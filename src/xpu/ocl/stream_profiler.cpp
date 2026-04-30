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

#include "xpu/ocl/context.hpp"
#include "xpu/ocl/stream_profiler.hpp"

#include "gpu/gpu_stream.hpp"

namespace dnnl {
namespace impl {
namespace xpu {
namespace ocl {

status_t stream_profiler_t::get_info(profiling_data_kind_t data_kind,
        int *num_entries, uint64_t *data) const {
    if (!num_entries) return status::invalid_arguments;
    bool is_per_kernel = (data_kind == profiling_data_kind::time_per_kernel);
    if (!data) {
        if (is_per_kernel) {
            *num_entries = (int)events_.size();
            return status::success;
        }
        std::unordered_set<uint64_t> seen;
        for (auto &ev : events_)
            seen.insert(ev.stamp);
        *num_entries = (int)seen.size();
        return status::success;
    }

    std::map<uint64_t, xpu::stream_profiler_t::entry_t> stamp2entry;
    int idx = 0;
    for (auto &ev : events_) {
        const xpu::ocl::event_t &ocl_event
                = *utils::downcast<xpu::ocl::event_t *>(ev.event.get());
        cl_ulong beg, end;
        assert(ocl_event.size() == 1);
        OCL_CHECK(xpu::ocl::clGetEventProfilingInfo(ocl_event[0].get(),
                CL_PROFILING_COMMAND_START, sizeof(beg), &beg, nullptr));
        OCL_CHECK(xpu::ocl::clGetEventProfilingInfo(ocl_event[0].get(),
                CL_PROFILING_COMMAND_END, sizeof(end), &end, nullptr));
        if (is_per_kernel) {
            data[idx++] = static_cast<uint64_t>(end - beg);
            continue;
        }
        auto &entry = stamp2entry[ev.stamp];
        entry.min_nsec = std::min(entry.min_nsec, beg);
        entry.max_nsec = std::max(entry.max_nsec, end);
        const auto *gpu_stream
                = utils::downcast<const gpu::stream_t *>(stream_);
        entry.freq += gpu_stream->get_freq(*ev.event);
        entry.kernel_count++;
    }
    if (is_per_kernel) return status::success;
    return xpu::stream_profiler_t::get_info_impl(stamp2entry, data_kind, data);
}

status_t stream_profiler_t::stop_async_event_polling() {
    if (!polling_active_.exchange(false)) {
        return status::success; // Already stopped
    }

    if (polling_thread_.joinable()) { polling_thread_.join(); }

    // Process any remaining pending events synchronously
    std::lock_guard<std::recursive_mutex> lock(m_);
    for (auto &event_info : pending_events_) {
        if (!event_info.out_evt) continue;

        // Convert xpu::event_t to cl_event for OpenCL wait
        const auto &ocl_event = xpu::ocl::event_t::from(*event_info.out_evt);
        if (ocl_event.size() > 0) {
            cl_event cl_ev = ocl_event[0].get();
            cl_int err = clWaitForEvents(1, &cl_ev);
            if (err == CL_SUCCESS) {
                double duration_ms = 0.0;
                get_aggregate_exec_timing(duration_ms, event_info.evt_snapshot);
                VPROF(event_info.start_ms, primitive, exec, VERBOSE_profile,
                        event_info.pd_info.c_str(), duration_ms);
            }
        }

        end_async_callback_tracking();
    }

    pending_events_.clear();
    return status::success;
}

status_t stream_profiler_t::get_aggregate_exec_timing(double &duration_ms,
        const std::vector<std::shared_ptr<xpu::event_t>> &evt_snap) const {

    duration_ms = 0.0;
    if (evt_snap.empty()) return status::success;

    cl_ulong agg_start = UINT64_MAX;
    cl_ulong agg_end = 0;

    for (const auto &ev : evt_snap) {
        if (!ev) continue;

        // Convert xpu::event_t to cl_event for OpenCL operations
        const auto &ocl_event = xpu::ocl::event_t::from(*ev);
        if (ocl_event.size() > 0) {
            cl_event cl_ev = ocl_event[0].get();

            cl_ulong evbeg, evend;
            OCL_CHECK(clGetEventProfilingInfo(cl_ev, CL_PROFILING_COMMAND_START,
                    sizeof(evbeg), &evbeg, nullptr));
            OCL_CHECK(clGetEventProfilingInfo(cl_ev, CL_PROFILING_COMMAND_END,
                    sizeof(evend), &evend, nullptr));
            agg_start = std::min(agg_start, evbeg);
            agg_end = std::max(agg_end, evend);
        }
    }

    if (agg_start != UINT64_MAX && agg_end > agg_start) {
        duration_ms = static_cast<double>(agg_end - agg_start) * 1e-6;
    }

    return status::success;
}

void stream_profiler_t::log_completed_primitive_events() {
    std::lock_guard<std::recursive_mutex> lock(m_);

    auto it = pending_events_.begin();
    while (it != pending_events_.end()) {
        if (!it->out_evt) {
            end_async_callback_tracking();
            it = pending_events_.erase(it);
            continue;
        }

        const auto &ocl_event = xpu::ocl::event_t::from(*it->out_evt);
        if (ocl_event.size() == 0) {
            end_async_callback_tracking();
            it = pending_events_.erase(it);
            continue;
        }

        cl_event cl_ev = ocl_event[0].get();
        cl_int event_status;
        cl_int err = clGetEventInfo(cl_ev, CL_EVENT_COMMAND_EXECUTION_STATUS,
                sizeof(event_status), &event_status, nullptr);

        if (err == CL_SUCCESS && event_status == CL_COMPLETE) {
            // Event completed - compute timing using stored event snapshot
            double duration_ms = 0.0;
            get_aggregate_exec_timing(duration_ms, it->evt_snapshot);
            VPROF(it->start_ms, primitive, exec, VERBOSE_profile,
                    it->pd_info.c_str(), duration_ms);

            end_async_callback_tracking();
            it = pending_events_.erase(it);

        } else if (err != CL_SUCCESS) {
            // gracefully exits upon profiler error
            VWARN(primitive, exec,
                    "%s, profiler error: failed to query event status",
                    it->pd_info.c_str());
            VPROF(it->start_ms, primitive, exec, VERBOSE_profile,
                    it->pd_info.c_str(), 0.f);

            end_async_callback_tracking();
            it = pending_events_.erase(it);
        } else {
            ++it;
        }
    }
}

} // namespace ocl
} // namespace xpu
} // namespace impl
} // namespace dnnl
