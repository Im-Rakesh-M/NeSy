import sys
import os
import asyncio
from typing import Dict, List, Callable, Any

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

class MessageBus:
    def __init__(self):
        self._subscribers: Dict[str, List[Callable[[Dict[str, Any]], asyncio.Future]]] = {}
        self._event_log: List[Dict] = []

    def subscribe(self, topic: str, callback: Callable[[Dict[str, Any]], Any]):
        if topic not in self._subscribers:
            self._subscribers[topic] = []
        self._subscribers[topic].append(callback)

        

    async def publish(self, topic: str, payload: Dict[str, Any]):
        if topic not in self._subscribers:
            return
        
        tasks = []
        for callback in self._subscribers[topic]:
            # Encapsulate as coroutine for asynchronous scheduling loop execution
            if asyncio.iscoroutinefunction(callback):
                tasks.append(asyncio.create_task(callback(payload)))
            else:
                # Fallback run within loop contextExecutor
                loop = asyncio.get_running_loop()
                tasks.append(loop.run_in_executor(None, callback, payload))
                
        if tasks:
            try:
                await asyncio.gather(*tasks, return_exceptions=True)
            except Exception:
                pass

    def get_topic_stats(self) -> dict:
        """
        Returns subscriber count per topic.

        Rationale: Allows the orchestrator to verify all
        agents have registered correctly before the sync
        loop starts. If a topic shows 0 subscribers, an
        agent failed to initialize — catch it early rather
        than silently dropping events at runtime.
        """
        return {
        topic: len(callbacks)
        for topic, callbacks in self._subscribers.items()
        }

    def get_event_count(self) -> int:
        """Returns total events published this session."""
        return len(self._event_log)