/*******************************************************************************
* Copyright 2023 Intel Corporation
*
* Licensed under the Apache License, Version 2.0 (the "License");
* you may not use this file except in compliance with the License.
* You may obtain a copy of the License at
*
*     http://www.apache.org/licenses/LICENSE-2.0
*
* Unless required by applicable law or agreed to in writing, software
* distributed under the License is distributed on an "AS IS" BASIS,
* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
* See the License for the specific language governing permissions and
* limitations under the License.
*******************************************************************************/

#include <CL/cl.h>

#include <map>
#include <unordered_set>

#include "common/c_types_map.hpp"
#include "common/utils.hpp"

#include "xpu/stream_profiler.hpp"

namespace dnnl {
namespace impl {
namespace xpu {

status_t stream_profiler_t::start_async_event_polling() {
    if (polling_active_.exchange(true)) { return status::success; }

    try {
        polling_thread_ = std::thread([this]() { polling_worker(); });
    } catch (const std::exception &e) {
        polling_active_.store(false);
        return status::runtime_error;
    }

    return status::success;
}

void stream_profiler_t::wait_for_async_event_completion() {
    std::unique_lock<std::recursive_mutex> lock(m_);
    async_completion_cv_.wait(
            lock, [this] { return async_tracking_count_.load() == 0; });
}

std::vector<std::shared_ptr<xpu::event_t>>
xpu::stream_profiler_t::extract_current_primitive_events() {
    std::unique_lock<std::recursive_mutex> lock(m_);
    std::vector<std::shared_ptr<xpu::event_t>> evt_snapshot;

    // Extract events for current stamp - generic implementation
    for (auto it = events_.rbegin(); it != events_.rend(); ++it) {
        if (it->stamp < stamp_) break;
        if (it->stamp == stamp_) { evt_snapshot.push_back(it->event->clone()); }
    }
    std::reverse(evt_snapshot.begin(), evt_snapshot.end());
    return evt_snapshot;
}

status_t xpu::stream_profiler_t::add_to_pending_async_event_list(
        std::shared_ptr<xpu::event_t> out_evt, double start_ms,
        const std::string &pd_info) {

    std::lock_guard<std::recursive_mutex> lock(m_);

    // Extract current primitive events using backend-specific implementation
    std::vector<std::shared_ptr<xpu::event_t>> evt_snapshot
            = extract_current_primitive_events();

    pending_events_.emplace_back(pending_async_event_t {
            out_evt, start_ms, pd_info, std::move(evt_snapshot)});

    start_async_callback_tracking();
    return status::success;
}

void stream_profiler_t::polling_worker() {
    const auto poll_interval
            = std::chrono::milliseconds(1); // 1ms polling interval
    const auto idle_interval = std::chrono::milliseconds(10); // 10ms when idle

    while (polling_active_.load()) {
        log_completed_primitive_events();

        // Check if there are pending events
        bool has_pending;
        {
            std::lock_guard<std::recursive_mutex> lock(m_);
            has_pending = !pending_events_.empty();
        }

        if (has_pending) {
            std::this_thread::sleep_for(poll_interval);
        } else {
            // No pending events, sleep longer or exit if not active
            if (polling_active_.load()) {
                std::this_thread::sleep_for(idle_interval);
            } else {
                break;
            }
        }
    }
}

} // namespace xpu
} // namespace impl
} // namespace dnnl
