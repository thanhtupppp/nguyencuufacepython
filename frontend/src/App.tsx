import { useState, useEffect, useCallback, useRef } from 'react';
import { Header } from './components/Header';
import { StatCards } from './components/StatCards';
import { CameraFeed } from './components/CameraFeed';
import { EventStream } from './components/EventStream';
import { EnrollModal } from './components/EnrollModal';
import { PersonsDrawer } from './components/PersonsDrawer';
import { useCamera } from './hooks/useCamera';
import { useWebSocket } from './hooks/useWebSocket';
import { checkHealth, fetchPersons, recognizeFace } from './services/api';
import { Person, RecognitionResult } from './types';

export function App() {
  // System statuses
  const [apiOnline, setApiOnline] = useState<boolean>(false);
  const { status: wsStatus, events, clearEvents } = useWebSocket();

  // Persons data
  const [persons, setPersons] = useState<Person[]>([]);

  // Modals & Drawers
  const [isEnrollOpen, setIsEnrollOpen] = useState(false);
  const [isPersonsOpen, setIsPersonsOpen] = useState(false);

  // Camera hook
  const {
    videoRef,
    isActive,
    isStreaming,
    error: cameraError,
    startCamera,
    stopCamera,
    captureFrameBlob,
  } = useCamera();

  // Auto scan & Recognition state
  const [autoScan, setAutoScan] = useState<boolean>(false);
  const [isScanning, setIsScanning] = useState<boolean>(false);
  const [lastResult, setLastResult] = useState<RecognitionResult | null>(null);
  const [inferenceTimeMs, setInferenceTimeMs] = useState<number | null>(null);

  // Scan statistics
  const [totalScans, setTotalScans] = useState<number>(0);
  const [matchedScans, setMatchedScans] = useState<number>(0);
  const [alertScans, setAlertScans] = useState<number>(0);

  // Load initial persons list
  const loadPersons = useCallback(async () => {
    try {
      const data = await fetchPersons();
      setPersons(data);
    } catch {
      // Ignored if API offline
    }
  }, []);

  // Health check polling
  useEffect(() => {
    let isMounted = true;
    const check = async () => {
      const healthy = await checkHealth();
      if (isMounted) {
        setApiOnline(healthy);
        if (healthy) loadPersons();
      }
    };

    check();
    const interval = setInterval(check, 10000);
    return () => {
      isMounted = false;
      clearInterval(interval);
    };
  }, [loadPersons]);

  // Execute Face Recognition
  const performRecognition = useCallback(async () => {
    if (isScanning || !isStreaming) return;

    try {
      setIsScanning(true);
      const blob = await captureFrameBlob(0.9);
      if (!blob) {
        setIsScanning(false);
        return;
      }

      const startTime = performance.now();
      const result = await recognizeFace(blob);
      const duration = Math.round(performance.now() - startTime);

      setInferenceTimeMs(duration);
      setLastResult(result);
      setTotalScans((prev) => prev + 1);

      if (result.status === 'MATCHED') {
        setMatchedScans((prev) => prev + 1);
      } else if (result.status !== 'NO_FACE') {
        setAlertScans((prev) => prev + 1);
      }
    } catch {
      // recognition error handled silently
    } finally {
      setIsScanning(false);
    }
  }, [isScanning, isStreaming, captureFrameBlob]);

  // Auto-scan timer loop (1.5s interval)
  const autoScanRef = useRef<number | null>(null);
  useEffect(() => {
    if (autoScan && isActive && isStreaming) {
      autoScanRef.current = window.setInterval(() => {
        performRecognition();
      }, 1500);
    } else {
      if (autoScanRef.current) {
        clearInterval(autoScanRef.current);
        autoScanRef.current = null;
      }
    }

    return () => {
      if (autoScanRef.current) {
        clearInterval(autoScanRef.current);
        autoScanRef.current = null;
      }
    };
  }, [autoScan, isActive, isStreaming, performRecognition]);

  return (
    <div className="min-h-screen bg-slate-950 text-slate-100 flex flex-col font-sans selection:bg-cyan-500 selection:text-white">
      {/* Top Navigation Bar */}
      <Header
        apiOnline={apiOnline}
        wsStatus={wsStatus}
        totalPersons={persons.length}
        onOpenPersons={() => setIsPersonsOpen(true)}
        onOpenEnroll={() => setIsEnrollOpen(true)}
      />

      {/* Main Content Dashboard */}
      <main className="flex-1 max-w-7xl w-full mx-auto px-4 lg:px-8 py-6">
        {/* KPI Metrics */}
        <StatCards
          totalPersons={persons.length}
          totalScans={totalScans}
          matchedScans={matchedScans}
          alertScans={alertScans}
        />

        {/* 2-Column Split: Camera & Event Feed */}
        <div className="grid grid-cols-1 lg:grid-cols-12 gap-6 items-start">
          {/* Left: Camera Feed & Real-time AI Reticle */}
          <div className="lg:col-span-7">
            <CameraFeed
              videoRef={videoRef}
              isActive={isActive}
              isStreaming={isStreaming}
              error={cameraError}
              onStartCamera={startCamera}
              onStopCamera={stopCamera}
              autoScan={autoScan}
              onToggleAutoScan={() => setAutoScan(!autoScan)}
              isScanning={isScanning}
              onManualScan={performRecognition}
              lastResult={lastResult}
              inferenceTimeMs={inferenceTimeMs}
            />
          </div>

          {/* Right: Real-time Event Stream via WebSocket */}
          <div className="lg:col-span-5">
            <EventStream events={events} onClear={clearEvents} />
          </div>
        </div>
      </main>

      {/* Slide-over Persons Drawer */}
      <PersonsDrawer
        isOpen={isPersonsOpen}
        onClose={() => setIsPersonsOpen(false)}
        persons={persons}
        onRefresh={loadPersons}
      />

      {/* Enroll Face Modal */}
      <EnrollModal
        isOpen={isEnrollOpen}
        onClose={() => setIsEnrollOpen(false)}
        onSuccess={() => {
          loadPersons();
        }}
      />
    </div>
  );
}

export default App;
