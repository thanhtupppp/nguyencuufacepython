import { useEffect, useRef, useState, useCallback } from 'react';
import { AccessEvent, ConnectionStatus } from '../types';

export function useWebSocket(url: string = 'ws://localhost:8000/ws/v1/events') {
  const [status, setStatus] = useState<ConnectionStatus>('connecting');
  const [events, setEvents] = useState<AccessEvent[]>([]);
  const [lastEvent, setLastEvent] = useState<AccessEvent | null>(null);
  const wsRef = useRef<WebSocket | null>(null);
  const reconnectTimeoutRef = useRef<number | null>(null);

  const clearEvents = useCallback(() => {
    setEvents([]);
  }, []);

  useEffect(() => {
    let isMounted = true;

    function connect() {
      if (!isMounted) return;
      setStatus('connecting');

      try {
        const ws = new WebSocket(url);
        wsRef.current = ws;

        ws.onopen = () => {
          if (!isMounted) return;
          setStatus('connected');
        };

        ws.onmessage = (event) => {
          if (!isMounted) return;
          try {
            const data: AccessEvent = JSON.parse(event.data);
            if (data.event_type === 'PONG') return;

            setLastEvent(data);
            if (data.event_type === 'ACCESS_EVENT' || data.event_type === 'SECURITY_ALERT') {
              setEvents((prev) => [
                { ...data, id: `${Date.now()}-${Math.random().toString(36).substr(2, 9)}` },
                ...prev.slice(0, 49), // Keep latest 50 events
              ]);
            }
          } catch {
            // Non-json or ping message
          }
        };

        ws.onclose = () => {
          if (!isMounted) return;
          setStatus('disconnected');
          // Reconnect in 3s
          reconnectTimeoutRef.current = window.setTimeout(connect, 3000);
        };

        ws.onerror = () => {
          ws.close();
        };
      } catch {
        setStatus('disconnected');
        reconnectTimeoutRef.current = window.setTimeout(connect, 3000);
      }
    }

    connect();

    // Heartbeat ping every 15 seconds
    const pingInterval = setInterval(() => {
      if (wsRef.current && wsRef.current.readyState === WebSocket.OPEN) {
        wsRef.current.send('ping');
      }
    }, 15000);

    return () => {
      isMounted = false;
      clearInterval(pingInterval);
      if (reconnectTimeoutRef.current) clearTimeout(reconnectTimeoutRef.current);
      if (wsRef.current) {
        wsRef.current.close();
      }
    };
  }, [url]);

  return { status, events, lastEvent, clearEvents };
}
