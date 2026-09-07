import React, { useState, useEffect } from 'react';
import { X, Sliders, Clock, Timer, Cpu, Check, RefreshCw, MapPin } from 'lucide-react';
import { DispenserConfig } from '../types';

interface DispenserSettingsModalProps {
  isOpen: boolean;
  onClose: () => void;
  config: DispenserConfig;
  onSave: (newConfig: DispenserConfig) => Promise<void>;
}

export const DispenserSettingsModal: React.FC<DispenserSettingsModalProps> = ({
  isOpen,
  onClose,
  config,
  onSave,
}) => {
  const [cooldownMinutes, setCooldownMinutes] = useState<number>(config.cooldown_minutes);
  const [dismissSeconds, setDismissSeconds] = useState<number>(config.dismiss_seconds);
  const [pulseMs, setPulseMs] = useState<number>(config.pulse_ms);
  const [deviceId, setDeviceId] = useState<string>(config.device_id);
  const [deviceName, setDeviceName] = useState<string>(config.device_name);

  const [isSaving, setIsSaving] = useState<boolean>(false);
  const [savedSuccess, setSavedSuccess] = useState<boolean>(false);

  // Sync state when config prop changes
  useEffect(() => {
    setCooldownMinutes(config.cooldown_minutes);
    setDismissSeconds(config.dismiss_seconds);
    setPulseMs(config.pulse_ms);
    setDeviceId(config.device_id);
    setDeviceName(config.device_name);
  }, [config, isOpen]);

  if (!isOpen) return null;

  const handleResetDefaults = () => {
    setCooldownMinutes(5.0);
    setDismissSeconds(3);
    setPulseMs(2500);
    setDeviceId('dispenser_01');
    setDeviceName('Máy Cấp Giấy Vệ Sinh #1');
  };

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    setIsSaving(true);
    try {
      await onSave({
        cooldown_minutes: Number(cooldownMinutes),
        dismiss_seconds: Number(dismissSeconds),
        pulse_ms: Number(pulseMs),
        device_id: deviceId.trim() || 'dispenser_01',
        device_name: deviceName.trim() || 'Máy Cấp Giấy Vệ Sinh',
      });
      setSavedSuccess(true);
      setTimeout(() => {
        setSavedSuccess(false);
        onClose();
      }, 1000);
    } catch {
      alert('Có lỗi xảy ra khi lưu cấu hình!');
    } finally {
      setIsSaving(false);
    }
  };

  const cooldownPresets = [
    { label: '1 phút (Test)', value: 1.0 },
    { label: '2 phút', value: 2.0 },
    { label: '3 phút', value: 3.0 },
    { label: '5 phút (Chuẩn)', value: 5.0 },
    { label: '10 phút', value: 10.0 },
    { label: '15 phút', value: 15.0 },
  ];

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center p-4 bg-slate-950/80 backdrop-blur-sm animate-in fade-in duration-200">
      <div className="bg-slate-900 border border-slate-800 w-full max-w-xl rounded-2xl shadow-2xl overflow-hidden flex flex-col max-h-[90vh]">
        {/* Header */}
        <div className="px-6 py-4 border-b border-slate-800 flex items-center justify-between">
          <div className="flex items-center space-x-2.5">
            <div className="p-2 rounded-xl bg-cyan-950/80 text-cyan-400 border border-cyan-800/40">
              <Sliders className="w-5 h-5" />
            </div>
            <div>
              <h3 className="font-bold text-white text-base">Cài Đặt Cấu Hình Máy Cấp Giấy</h3>
              <p className="text-xs text-slate-400">Tùy chỉnh thời gian chống lạm dụng & thông số máy</p>
            </div>
          </div>
          <button
            onClick={onClose}
            className="p-1.5 rounded-lg text-slate-400 hover:text-white hover:bg-slate-800 transition"
          >
            <X className="w-5 h-5" />
          </button>
        </div>

        {/* Settings Form */}
        <form onSubmit={handleSubmit} className="p-6 space-y-5 overflow-y-auto flex-1">
          {/* SECTION 1: ANTI-ABUSE COOLDOWN DURATION */}
          <div className="space-y-2.5">
            <div className="flex items-center justify-between">
              <label className="text-xs font-bold text-slate-200 uppercase tracking-wide flex items-center space-x-1.5">
                <Clock className="w-4 h-4 text-cyan-400" />
                <span>Thời Gian Chờ Chống Lạm Dụng (Cooldown)</span>
              </label>
              <span className="px-2.5 py-0.5 rounded-full text-xs font-mono font-bold bg-cyan-950 border border-cyan-800 text-cyan-300">
                {cooldownMinutes} Phút
              </span>
            </div>

            {/* Quick Presets */}
            <div className="grid grid-cols-3 sm:grid-cols-6 gap-1.5">
              {cooldownPresets.map((preset) => (
                <button
                  key={preset.value}
                  type="button"
                  onClick={() => setCooldownMinutes(preset.value)}
                  className={`py-1.5 px-2 rounded-lg text-xs font-medium border transition ${
                    cooldownMinutes === preset.value
                      ? 'bg-cyan-600 border-cyan-500 text-white shadow-sm'
                      : 'bg-slate-950 border-slate-800 text-slate-400 hover:text-white hover:border-slate-700'
                  }`}
                >
                  {preset.label}
                </button>
              ))}
            </div>

            {/* Range Slider & Manual Input */}
            <div className="flex items-center space-x-3 pt-1">
              <input
                type="range"
                min="0.5"
                max="30"
                step="0.5"
                value={cooldownMinutes}
                onChange={(e) => setCooldownMinutes(parseFloat(e.target.value))}
                className="flex-1 accent-cyan-500 h-2 bg-slate-950 rounded-lg cursor-pointer"
              />
              <div className="w-20 flex items-center bg-slate-950 border border-slate-800 rounded-lg px-2 py-1">
                <input
                  type="number"
                  min="0.5"
                  max="1440"
                  step="0.5"
                  value={cooldownMinutes}
                  onChange={(e) => setCooldownMinutes(parseFloat(e.target.value) || 1)}
                  className="w-full bg-transparent text-xs text-white focus:outline-none text-right font-mono font-bold"
                />
                <span className="text-[10px] text-slate-500 ml-1">phút</span>
              </div>
            </div>

            <p className="text-[11px] text-slate-400">
              💡 Mỗi người chỉ được lấy giấy 1 lần trong khoảng <strong>{cooldownMinutes} phút</strong>. Lần lấy tiếp theo trong khoảng này sẽ bị từ chối và hiện đồng hồ đếm ngược.
            </p>
          </div>

          <div className="border-t border-slate-800/80" />

          {/* SECTION 2: AUTO DISMISS TIMER */}
          <div className="space-y-2">
            <div className="flex items-center justify-between">
              <label className="text-xs font-bold text-slate-200 uppercase tracking-wide flex items-center space-x-1.5">
                <Timer className="w-4 h-4 text-cyan-400" />
                <span>Thời Gian Tự Động Đóng Thông Báo</span>
              </label>
              <span className="text-xs font-mono font-bold text-cyan-400">
                {dismissSeconds} Giây
              </span>
            </div>

            <div className="flex items-center space-x-2">
              {[2, 3, 5, 8].map((sec) => (
                <button
                  key={sec}
                  type="button"
                  onClick={() => setDismissSeconds(sec)}
                  className={`flex-1 py-1.5 rounded-lg text-xs font-medium border transition ${
                    dismissSeconds === sec
                      ? 'bg-cyan-600 border-cyan-500 text-white'
                      : 'bg-slate-950 border-slate-800 text-slate-400 hover:text-white'
                  }`}
                >
                  {sec} Giây {sec === 3 ? '(Chuẩn)' : ''}
                </button>
              ))}
            </div>
            <p className="text-[11px] text-slate-400">
              Bảng thông báo kết quả sẽ hiển thị trong {dismissSeconds}s rồi tự quay về màn hình chờ quét cho người tiếp theo.
            </p>
          </div>

          <div className="border-t border-slate-800/80" />

          {/* SECTION 3: MOTOR PULSE DURATION */}
          <div className="space-y-2">
            <div className="flex items-center justify-between">
              <label className="text-xs font-bold text-slate-200 uppercase tracking-wide flex items-center space-x-1.5">
                <Cpu className="w-4 h-4 text-cyan-400" />
                <span>Thời Gian Xung Động Cơ (Lượng giấy ra)</span>
              </label>
              <span className="text-xs font-mono font-bold text-cyan-400">
                {(pulseMs / 1000).toFixed(1)} Giây
              </span>
            </div>

            <div className="flex items-center space-x-2">
              {[
                { label: '1.5s (Ít giấy)', ms: 1500 },
                { label: '2.5s (Tiêu chuẩn)', ms: 2500 },
                { label: '3.5s (Nhiều giấy)', ms: 3500 },
              ].map((item) => (
                <button
                  key={item.ms}
                  type="button"
                  onClick={() => setPulseMs(item.ms)}
                  className={`flex-1 py-1.5 rounded-lg text-xs font-medium border transition ${
                    pulseMs === item.ms
                      ? 'bg-cyan-600 border-cyan-500 text-white'
                      : 'bg-slate-950 border-slate-800 text-slate-400 hover:text-white'
                  }`}
                >
                  {item.label}
                </button>
              ))}
            </div>
            <p className="text-[11px] text-slate-400">
              Độ dài tín hiệu xung WebSocket `pulse_ms` gửi đến ESP32 / Relay điều khiển motor cuộn giấy.
            </p>
          </div>

          <div className="border-t border-slate-800/80" />

          {/* SECTION 4: DEVICE INFO */}
          <div className="space-y-2">
            <label className="text-xs font-bold text-slate-200 uppercase tracking-wide flex items-center space-x-1.5">
              <MapPin className="w-4 h-4 text-cyan-400" />
              <span>Tên Thiết Bị & Vị Trí Lắp Đặt</span>
            </label>
            <div className="grid grid-cols-2 gap-3">
              <div>
                <span className="text-[11px] text-slate-400 block mb-1">Mã thiết bị</span>
                <input
                  type="text"
                  value={deviceId}
                  onChange={(e) => setDeviceId(e.target.value)}
                  className="w-full bg-slate-950 border border-slate-800 rounded-xl px-3 py-2 text-xs text-white focus:outline-none focus:border-cyan-500"
                />
              </div>
              <div>
                <span className="text-[11px] text-slate-400 block mb-1">Tên / Vị trí đặt máy</span>
                <input
                  type="text"
                  value={deviceName}
                  onChange={(e) => setDeviceName(e.target.value)}
                  className="w-full bg-slate-950 border border-slate-800 rounded-xl px-3 py-2 text-xs text-white focus:outline-none focus:border-cyan-500"
                />
              </div>
            </div>
          </div>

          {/* Action Buttons */}
          <div className="pt-3 border-t border-slate-800 flex items-center justify-between">
            <button
              type="button"
              onClick={handleResetDefaults}
              className="text-xs text-slate-500 hover:text-slate-300 transition"
            >
              Khôi phục mặc định
            </button>

            <div className="flex items-center space-x-2">
              <button
                type="button"
                onClick={onClose}
                className="px-4 py-2 rounded-xl text-xs font-medium text-slate-400 hover:text-white transition"
              >
                Hủy
              </button>
              <button
                type="submit"
                disabled={isSaving}
                className="px-5 py-2 rounded-xl text-xs font-semibold bg-cyan-600 hover:bg-cyan-500 text-white shadow-md shadow-cyan-600/30 flex items-center space-x-1.5 transition active:scale-95"
              >
                {isSaving ? (
                  <RefreshCw className="w-3.5 h-3.5 animate-spin" />
                ) : savedSuccess ? (
                  <Check className="w-3.5 h-3.5 text-white" />
                ) : null}
                <span>{savedSuccess ? 'Đã lưu!' : isSaving ? 'Đang lưu...' : 'Lưu Cài Đặt'}</span>
              </button>
            </div>
          </div>
        </form>
      </div>
    </div>
  );
};
