import { useState, useEffect, useCallback } from 'react';
import { Header } from './components/Header';
import { StatCards } from './components/StatCards';
import { CameraFeed } from './components/CameraFeed';
import { EventStream } from './components/EventStream';
import { EnrollModal } from './components/EnrollModal';
import { PersonsDrawer } from './components/PersonsDrawer';
import { DispenserKiosk } from './components/DispenserKiosk';
import { DispenserLogs } from './components/DispenserLogs';
import { DispenserSettingsModal } from './components/DispenserSettingsModal';
import { useCamera } from './hooks/useCamera';
import { useWebSocket } from './hooks/useWebSocket';
import {
  checkHealth,
  fetchPersons,
  recognizeFace,
  requestToiletPaper,
  fetchDispenserStats,
  fetchDispenserLogs,
  fetchDispenserConfig,
  updateDispenserConfig,
} from './services/api';
import { Person, RecognitionResult, DispenseResult, DispenserStats, DispenseLog, DispenserConfig } from './types';
import { Shield, Sparkles } from 'lucide-react';

export function App() {
  // App view mode: 'dispenser' (Smart Toilet Paper Kiosk) or 'admin' (Security Surveillance & Management)
  const [appMode, setAppMode] = useState<'dispenser' | 'admin'>('dispenser');

  // System statuses
  const [apiOnline, setApiOnline] = useState<boolean>(false);
  const { status: wsStatus, events, lastEvent, clearEvents } = useWebSocket();

  // Persons data
  const [persons, setPersons] = useState<Person[]>([]);

  // Dispenser state
  const [cooldownMinutes, setCooldownMinutes] = useState<number>(5.0);
  const [dispenserConfig, setDispenserConfig] = useState<DispenserConfig>({
    cooldown_minutes: 5.0,
    dismiss_seconds: 3,
    pulse_ms: 2500,
    device_id: 'dispenser_01',
    device_name: 'Máy Cấp Giấy Vệ Sinh #1',
  });
  const [dispenserStats, setDispenserStats] = useState<DispenserStats | null>(null);
  const [dispenserLogs, setDispenserLogs] = useState<DispenseLog[]>([]);
  const [isDispensing, setIsDispensing] = useState<boolean>(false);

  // Modals & Drawers
  const [isEnrollOpen, setIsEnrollOpen] = useState(false);
  const [isPersonsOpen, setIsPersonsOpen] = useState(false);
  const [isSettingsOpen, setIsSettingsOpen] = useState(false);

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

  // Admin Surveillance / Auto-scan state
  const [autoScan, setAutoScan] = useState<boolean>(false);
  const [isScanning, setIsScanning] = useState<boolean>(false);
  const [lastResult, setLastResult] = useState<RecognitionResult | null>(null);
  const [inferenceTimeMs, setInferenceTimeMs] = useState<number | null>(null);

  // Admin scan statistics
  const [totalScans, setTotalScans] = useState<number>(0);
  const [matchedScans, setMatchedScans] = useState<number>(0);
  const [alertScans, setAlertScans] = useState<number>(0);

  // Load persons list
  const loadPersons = useCallback(async () => {
    try {
      const data = await fetchPersons();
      setPersons(data);
    } catch {
      // Ignored if API offline
    }
  }, []);

  // Load dispenser stats, logs & config
  const loadDispenserData = useCallback(async () => {
    try {
      const [stats, logs, cfg] = await Promise.all([
        fetchDispenserStats(),
        fetchDispenserLogs(30),
        fetchDispenserConfig().catch(() => null),
      ]);
      setDispenserStats(stats);
      setDispenserLogs(logs);
      if (cfg) {
        setDispenserConfig(cfg);
        setCooldownMinutes(cfg.cooldown_minutes);
      }
    } catch {
      // Ignored if API offline
    }
  }, []);

  const handleSaveConfig = async (newConfig: DispenserConfig) => {
    const updated = await updateDispenserConfig(newConfig);
    setDispenserConfig(updated);
    setCooldownMinutes(updated.cooldown_minutes);
  };

  // Health check polling
  useEffect(() => {
    let isMounted = true;
    const check = async () => {
      const healthy = await checkHealth();
      if (isMounted) {
        setApiOnline(healthy);
        if (healthy) {
          loadPersons();
          loadDispenserData();
        }
      }
    };

    check();
    const interval = setInterval(check, 10000);
    return () => {
      isMounted = false;
      clearInterval(interval);
    };
  }, [loadPersons, loadDispenserData]);

  // Update dispenser logs upon WebSocket trigger
  useEffect(() => {
    if (!lastEvent) return;
    if (lastEvent.event_type === 'DISPENSER_TRIGGER' || lastEvent.event_type === 'DISPENSER_BLOCKED') {
      loadDispenserData();
    }
  }, [lastEvent, loadDispenserData]);

  // Auto-start camera when mounting Kiosk mode
  useEffect(() => {
    if (!isActive && !cameraError) {
      startCamera();
    }
  }, [isActive, cameraError, startCamera]);

  // Handle Request Toilet Paper from Kiosk
  const handleRequestPaper = async (cooldownMin: number): Promise<DispenseResult | null> => {
    if (isDispensing || !isStreaming) return null;

    try {
      setIsDispensing(true);
      const blob = await captureFrameBlob(0.92);
      if (!blob) {
        setIsDispensing(false);
        return {
          granted: false,
          status: 'NO_FRAME',
          message: 'Không thể chụp ảnh từ camera. Vui lòng thử lại!',
          cooldown_remaining_seconds: 0,
        };
      }

      const res = await requestToiletPaper(blob, cooldownMin);
      await loadDispenserData();
      await loadPersons();
      return res;
    } catch (err: any) {
      return {
        granted: false,
        status: 'ERROR',
        message: err?.message || 'Có lỗi khi liên lạc với máy chủ!',
        cooldown_remaining_seconds: 0,
      };
    } finally {
      setIsDispensing(false);
    }
  };

  // Execute Face Recognition for Admin Surveillance
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

  // Admin Auto-scan timer loop (1.5s interval)
  useEffect(() => {
    let intervalId: number | null = null;
    if (appMode === 'admin' && autoScan && isActive && isStreaming) {
      intervalId = window.setInterval(() => {
        performRecognition();
      }, 1500);
    }
    return () => {
      if (intervalId) clearInterval(intervalId);
    };
  }, [appMode, autoScan, isActive, isStreaming, performRecognition]);

  return (
    <div className="min-h-screen bg-slate-950 text-slate-100 flex flex-col font-sans selection:bg-cyan-500 selection:text-white">
      {/* Header */}
      <Header
        apiOnline={apiOnline}
        wsStatus={wsStatus}
        totalPersons={persons.length}
        onOpenPersons={() => setIsPersonsOpen(true)}
        onOpenEnroll={() => setIsEnrollOpen(true)}
        onOpenSettings={() => setIsSettingsOpen(true)}
      />

      {/* Mode Switcher Banner */}
      <div className="bg-slate-900/60 border-b border-slate-800/80 px-4 py-2">
        <div className="max-w-7xl mx-auto flex items-center justify-between">
          <div className="flex items-center space-x-1 bg-slate-950 p-1 rounded-xl border border-slate-800">
            <button
              onClick={() => setAppMode('dispenser')}
              className={`flex items-center space-x-2 px-4 py-1.5 rounded-lg text-xs font-semibold transition ${
                appMode === 'dispenser'
                  ? 'bg-gradient-to-r from-cyan-600 to-blue-600 text-white shadow-md shadow-cyan-600/30'
                  : 'text-slate-400 hover:text-slate-200'
              }`}
            >
              <span>🧻 Kiosk Cấp Giấy Vệ Sinh</span>
              <span className="px-1.5 py-0.2 bg-cyan-950/80 text-cyan-300 border border-cyan-800 text-[10px] rounded">
                Chính
              </span>
            </button>

            <button
              onClick={() => setAppMode('admin')}
              className={`flex items-center space-x-2 px-4 py-1.5 rounded-lg text-xs font-semibold transition ${
                appMode === 'admin'
                  ? 'bg-slate-800 text-white shadow-sm border border-slate-700'
                  : 'text-slate-400 hover:text-slate-200'
              }`}
            >
              <Shield className="w-3.5 h-3.5" />
              <span>Chế Độ Giám Sát & CCTV</span>
            </button>
          </div>

          <div className="hidden md:flex items-center space-x-2 text-xs text-slate-400">
            <Sparkles className="w-3.5 h-3.5 text-yellow-400" />
            <span>Tự động nhận diện & chống lạm dụng Cooldown</span>
          </div>
        </div>
      </div>

      {/* Main Content Dashboard */}
      <main className="flex-1 max-w-7xl w-full mx-auto px-4 lg:px-8 py-6">
        {appMode === 'dispenser' ? (
          // =========================================================
          // SMART TOILET PAPER DISPENSER KIOSK VIEW
          // =========================================================
          <div className="grid grid-cols-1 lg:grid-cols-12 gap-6 items-start">
            {/* Left: Interactive Touchscreen Kiosk */}
            <div className="lg:col-span-7">
              <DispenserKiosk
                videoRef={videoRef}
                isActive={isActive}
                isStreaming={isStreaming}
                cameraError={cameraError}
                onStartCamera={startCamera}
                onStopCamera={stopCamera}
                onRequestPaper={handleRequestPaper}
                isProcessing={isDispensing}
                cooldownMinutes={cooldownMinutes}
                onChangeCooldown={(min) => {
                  setCooldownMinutes(min);
                  handleSaveConfig({ ...dispenserConfig, cooldown_minutes: min });
                }}
                onOpenSettings={() => setIsSettingsOpen(true)}
                viewfinderStyle={dispenserConfig.viewfinder_style || 'hud'}
                voiceEnabled={dispenserConfig.voice_enabled ?? true}
                voiceVolume={dispenserConfig.voice_volume ?? 1.0}
                voiceRate={dispenserConfig.voice_rate ?? 1.0}
                touchlessEnabled={dispenserConfig.touchless_enabled ?? true}
                touchlessDelay={dispenserConfig.touchless_delay ?? 1.5}
                welcomeVoiceEnabled={dispenserConfig.welcome_voice_enabled ?? true}
              />
            </div>

            {/* Right: Live Dispenser Logs & Anti-Abuse Metrics */}
            <div className="lg:col-span-5">
              <DispenserLogs
                logs={dispenserLogs}
                stats={dispenserStats}
                cooldownMinutes={cooldownMinutes}
                onClear={() => setDispenserLogs([])}
                onRefresh={loadDispenserData}
              />
            </div>
          </div>
        ) : (
          // =========================================================
          // ADMIN SURVEILLANCE & SCANNER VIEW
          // =========================================================
          <>
            <StatCards
              totalPersons={persons.length}
              totalScans={totalScans}
              matchedScans={matchedScans}
              alertScans={alertScans}
            />

            <div className="grid grid-cols-1 lg:grid-cols-12 gap-6 items-start">
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

              <div className="lg:col-span-5">
                <EventStream events={events} onClear={clearEvents} />
              </div>
            </div>
          </>
        )}
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
          loadDispenserData();
        }}
      />

      {/* Dispenser Settings Modal */}
      <DispenserSettingsModal
        isOpen={isSettingsOpen}
        onClose={() => setIsSettingsOpen(false)}
        config={dispenserConfig}
        onSave={handleSaveConfig}
      />
    </div>
  );
}

export default App;
