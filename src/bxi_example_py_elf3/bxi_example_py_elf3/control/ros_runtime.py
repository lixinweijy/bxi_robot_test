"""Keep the ROS context alive until controller callbacks have stopped."""

import signal

import rclpy
from rclpy.executors import ExternalShutdownException, SingleThreadedExecutor
from rclpy.signals import SignalHandlerOptions


def run_controller(node_factory, args=None):
    node = None
    executor = None
    stop_requested = False

    def request_stop(_signum, _frame):
        nonlocal stop_requested
        stop_requested = True

    previous_handlers = {
        sig: signal.signal(sig, request_stop)
        for sig in (signal.SIGINT, signal.SIGTERM)
    }
    try:
        # Keep the context valid until the current callback and cleanup finish.
        rclpy.init(args=args, signal_handler_options=SignalHandlerOptions.NO)
        node = node_factory()
        executor = SingleThreadedExecutor()
        executor.add_node(node)
        while (rclpy.ok() and not stop_requested
               and not node.shutdown_requested.is_set()):
            executor.spin_once(timeout_sec=0.05)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        try:
            if node is not None:
                node.shutdown_requested.set()
                for timer in node.timers:
                    timer.cancel()
            if executor is not None:
                executor.shutdown()
        finally:
            try:
                if node is not None:
                    node.destroy_node()
            finally:
                try:
                    rclpy.try_shutdown()
                finally:
                    for sig, handler in previous_handlers.items():
                        signal.signal(sig, handler)
