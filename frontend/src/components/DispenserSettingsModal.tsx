import React, { useState, useEffect } from 'react';
import { X, Sliders, Clock, Timer, Cpu, Check, RefreshCw, MapPin, ScanFace, Volume2, VolumeX, Play, Hand, Zap } from 'lucide-react';
import { DispenserConfig } from '../types';
import { soundEffects } from '../utils/audio';

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
  const [viewfinderStyle, setViewfinderStyle] = useState<'hud' | 'corners' | 'oval'>(
    config.viewfinder_style || 'hud'
  );
  const [voiceEnabled, setVoiceEnabled] = useState<boolean>(config.voice_enabled ?? true);
  const [voiceVolume, setVoiceVolume] = useState<number>(config.voice_volume ?? 1.0);
  const [voiceRate, setVoiceRate] = useState<number>(config.voice_rate ?? 1.0);
  const [touchlessEnabled, setTouchlessEnabled] = useState<boolean>(config.touchless_enabled ?? true);
  const [touchlessDelay, setTouchlessDelay] = useState<number>(config.touchless_delay ?? 1.5);
  const [welcomeVoiceEnabled, setWelcomeVoiceEnabled] = useState<boolean>(config.welcome_voice_enabled ?? true);

  const [isSaving, setIsSaving] = useState<boolean>(false);
  const [savedSuccess, setSavedSuccess] = useState<boolean>(false);

  // Sync state when config prop changes
  useEffect(() => {
    setCooldownMinutes(config.cooldown_minutes);
    setDismissSeconds(config.dismiss_seconds);
    setPulseMs(config.pulse_ms);
    setDeviceId(config.device_id);
    setDeviceName(config.device_name);
    setViewfinderStyle(config.viewfinder_style || 'hud');
    setVoiceEnabled(config.voice_enabled ?? true);
    setVoiceVolume(config.voice_volume ?? 1.0);
    setVoiceRate(config.voice_rate ?? 1.0);
    setTouchlessEnabled(config.touchless_enabled ?? true);
    setTouchlessDelay(config.touchless_delay ?? 1.5);
    setWelcomeVoiceEnabled(config.welcome_voice_enabled ?? true);
  }, [config, isOpen]);

  if (!isOpen) return null;

  const handleResetDefaults = () => {
    setCooldownMinutes(5.0);
    setDismissSeconds(3);
    setPulseMs(2500);
    setDeviceId('dispenser_01');
    setDeviceName('Máy Cấp Giấy Vệ Sinh #1');
    setViewfinderStyle('hud');
    setVoiceEnabled(true);
    setVoiceVolume(1.0);
    setVoiceRate(1.0);
    setTouchlessEnabled(true);
    setTouchlessDelay(1.5);
    setWelcomeVoiceEnabled(true);
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
        viewfinder_style: viewfinderStyle,
        voice_enabled: voiceEnabled,
        voice_volume: Number(voiceVolume),
        voice_rate: Number(voiceRate),
        touchless_enabled: touchlessEnabled,
        touchless_delay: Number(touchlessDelay),
        welcome_voice_enabled: welcomeVoiceEnabled,
      });
      soundEffects.setVoiceConfig(voiceEnabled, voiceVolume, voiceRate);
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

          {/* SECTION 4: VIEWFINDER HUD STYLE */}
          <div className="space-y-2.5">
            <div className="flex items-center justify-between">
              <label className="text-xs font-bold text-slate-200 uppercase tracking-wide flex items-center space-x-1.5">
                <ScanFace className="w-4 h-4 text-cyan-400" />
                <span>Kiểu Khung Ngắm Nhận Diện (Viewfinder Style)</span>
              </label>
              <span className="text-xs font-mono font-bold text-cyan-400 uppercase">
                {viewfinderStyle}
              </span>
            </div>

            <div className="grid grid-cols-1 sm:grid-cols-3 gap-2">
              <button
                type="button"
                onClick={() => setViewfinderStyle('hud')}
                className={`p-3 rounded-xl border text-left transition flex flex-col justify-between space-y-1.5 ${
                  viewfinderStyle === 'hud'
                    ? 'bg-cyan-950/60 border-cyan-500 text-cyan-200 shadow-md shadow-cyan-950/40'
                    : 'bg-slate-950 border-slate-800 text-slate-400 hover:text-white hover:border-slate-700'
                }`}
              >
                <div className="flex items-center justify-between w-full">
                  <span className="text-xs font-bold">🎯 AI HUD Cao Cấp</span>
                  {viewfinderStyle === 'hud' && <Check className="w-3.5 h-3.5 text-cyan-400" />}
                </div>
                <p className="text-[10px] text-slate-400 leading-tight">
                  Mờ tối viền ngoài, 4 góc ngắm công nghệ + viền oval phát sáng + vạch chữ thập.
                </p>
              </button>

              <button
                type="button"
                onClick={() => setViewfinderStyle('corners')}
                className={`p-3 rounded-xl border text-left transition flex flex-col justify-between space-y-1.5 ${
                  viewfinderStyle === 'corners'
                    ? 'bg-cyan-950/60 border-cyan-500 text-cyan-200 shadow-md shadow-cyan-950/40'
                    : 'bg-slate-950 border-slate-800 text-slate-400 hover:text-white hover:border-slate-700'
                }`}
              >
                <div className="flex items-center justify-between w-full">
                  <span className="text-xs font-bold">🔲 Góc Ngắm AI</span>
                  {viewfinderStyle === 'corners' && <Check className="w-3.5 h-3.5 text-cyan-400" />}
                </div>
                <p className="text-[10px] text-slate-400 leading-tight">
                  4 góc ngắm L-bracket tối giản, trong suốt 100%, không che viền hậu cảnh.
                </p>
              </button>

              <button
                type="button"
                onClick={() => setViewfinderStyle('oval')}
                className={`p-3 rounded-xl border text-left transition flex flex-col justify-between space-y-1.5 ${
                  viewfinderStyle === 'oval'
                    ? 'bg-cyan-950/60 border-cyan-500 text-cyan-200 shadow-md shadow-cyan-950/40'
                    : 'bg-slate-950 border-slate-800 text-slate-400 hover:text-white hover:border-slate-700'
                }`}
              >
                <div className="flex items-center justify-between w-full">
                  <span className="text-xs font-bold">⭕ Apple FaceID</span>
                  {viewfinderStyle === 'oval' && <Check className="w-3.5 h-3.5 text-cyan-400" />}
                </div>
                <p className="text-[10px] text-slate-400 leading-tight">
                  Vòng sinh trắc học đồng tâm mượt mà, phát sáng quang học khi quét.
                </p>
              </button>
            </div>
          </div>

          <div className="border-t border-slate-800/80" />

          {/* SECTION 5: VOICE ANNOUNCEMENTS */}
          <div className="space-y-3">
            <div className="flex items-center justify-between">
              <label className="text-xs font-bold text-slate-200 uppercase tracking-wide flex items-center space-x-1.5">
                <Volume2 className="w-4 h-4 text-cyan-400" />
                <span>Thông Báo Bằng Giọng Nói Tiếng Việt</span>
              </label>
              <button
                type="button"
                onClick={() => setVoiceEnabled(!voiceEnabled)}
                className={`flex items-center space-x-1.5 px-3 py-1 rounded-full text-xs font-semibold transition ${
                  voiceEnabled
                    ? 'bg-cyan-600 text-white shadow-sm'
                    : 'bg-slate-800 text-slate-400'
                }`}
              >
                {voiceEnabled ? <Volume2 className="w-3.5 h-3.5" /> : <VolumeX className="w-3.5 h-3.5" />}
                <span>{voiceEnabled ? 'ĐANG BẬT' : 'ĐÃ TẮT'}</span>
              </button>
            </div>

            {voiceEnabled && (
              <div className="bg-slate-950 p-3 rounded-xl border border-slate-800 space-y-3 animate-in fade-in duration-150">
                <div className="flex items-center justify-between">
                  <span className="text-[11px] text-slate-300">Nghe thử giọng nói:</span>
                  <button
                    type="button"
                    onClick={() => {
                      soundEffects.setVoiceConfig(true, voiceVolume, voiceRate);
                      soundEffects.testVoice();
                    }}
                    className="px-3 py-1 bg-cyan-950 hover:bg-cyan-900 border border-cyan-700 text-cyan-300 rounded-lg text-xs font-medium flex items-center space-x-1.5 transition active:scale-95"
                  >
                    <Play className="w-3 h-3 fill-cyan-400 text-cyan-400" />
                    <span>Phát Mẫu Thử</span>
                  </button>
                </div>

                <div className="grid grid-cols-2 gap-3 pt-1">
                  <div>
                    <div className="flex justify-between text-[10px] text-slate-400 mb-1">
                      <span>Âm lượng giọng:</span>
                      <span className="text-cyan-400 font-mono font-bold">{Math.round(voiceVolume * 100)}%</span>
                    </div>
                    <input
                      type="range"
                      min="0.1"
                      max="1.0"
                      step="0.1"
                      value={voiceVolume}
                      onChange={(e) => setVoiceVolume(parseFloat(e.target.value))}
                      className="w-full accent-cyan-500 h-1.5 bg-slate-900 rounded-lg cursor-pointer"
                    />
                  </div>
                  <div>
                    <div className="flex justify-between text-[10px] text-slate-400 mb-1">
                      <span>Tốc độ đọc:</span>
                      <span className="text-cyan-400 font-mono font-bold">{voiceRate.toFixed(1)}x</span>
                    </div>
                    <input
                      type="range"
                      min="0.7"
                      max="1.3"
                      step="0.1"
                      value={voiceRate}
                      onChange={(e) => setVoiceRate(parseFloat(e.target.value))}
                      className="w-full accent-cyan-500 h-1.5 bg-slate-900 rounded-lg cursor-pointer"
                    />
                  </div>
                </div>

                <p className="text-[10px] text-slate-500 italic">
                  💡 Giọng nói sẽ tự động phát khi: Cấp giấy thành công, nhắc nhở chờ Cooldown, nhắc tháo khẩu trang/bỏ tay che mặt hoặc cảnh báo lỗi.
                </p>
              </div>
            )}
          </div>

          <div className="border-t border-slate-800/80" />

          {/* SECTION 6: SMART PRESENCE & TOUCHLESS MODE */}
          <div className="space-y-3">
            <div className="flex items-center justify-between">
              <label className="text-xs font-bold text-slate-200 uppercase tracking-wide flex items-center space-x-1.5">
                <Zap className="w-4 h-4 text-emerald-400" />
                <span>Cấp Giấy Tự Động Không Chạm (100% Touchless)</span>
              </label>
              <button
                type="button"
                onClick={() => setTouchlessEnabled(!touchlessEnabled)}
                className={`flex items-center space-x-1.5 px-3 py-1 rounded-full text-xs font-semibold transition ${
                  touchlessEnabled
                    ? 'bg-emerald-600 text-white shadow-sm'
                    : 'bg-slate-800 text-slate-400'
                }`}
              >
                <Hand className="w-3.5 h-3.5" />
                <span>{touchlessEnabled ? 'TỰ ĐỘNG BẬT' : 'BẤM NÚT TAY'}</span>
              </button>
            </div>

            {touchlessEnabled ? (
              <div className="bg-slate-950 p-3 rounded-xl border border-slate-800 space-y-3 animate-in fade-in duration-150">
                <div className="flex items-center justify-between">
                  <span className="text-[11px] text-slate-300">Thời gian giữ mặt trước khi nhả giấy:</span>
                  <span className="text-xs font-mono font-bold text-emerald-400">
                    {touchlessDelay} Giây
                  </span>
                </div>

                <div className="grid grid-cols-4 gap-2">
                  {[
                    { label: '1.0s (Nhanh)', val: 1.0 },
                    { label: '1.5s (Chuẩn)', val: 1.5 },
                    { label: '2.0s (Kỹ)', val: 2.0 },
                    { label: '2.5s (Chậm)', val: 2.5 },
                  ].map((item) => (
                    <button
                      key={item.val}
                      type="button"
                      onClick={() => setTouchlessDelay(item.val)}
                      className={`py-1.5 rounded-lg text-xs font-medium border transition ${
                        touchlessDelay === item.val
                          ? 'bg-emerald-600 border-emerald-500 text-white'
                          : 'bg-slate-900 border-slate-800 text-slate-400 hover:text-white'
                      }`}
                    >
                      {item.label}
                    </button>
                  ))}
                </div>

                {/* Welcome Voice Greeting Toggle */}
                <div className="pt-2 border-t border-slate-800/80 flex items-center justify-between">
                  <div>
                    <span className="text-xs text-slate-200 block font-medium">Giọng chào khi có người bước tới</span>
                    <span className="text-[10px] text-slate-400">"Xin chào bạn! Vui lòng nhìn thẳng camera..."</span>
                  </div>
                  <button
                    type="button"
                    onClick={() => setWelcomeVoiceEnabled(!welcomeVoiceEnabled)}
                    className={`px-2.5 py-1 rounded-lg text-[11px] font-semibold border transition ${
                      welcomeVoiceEnabled
                        ? 'bg-cyan-950 border-cyan-700 text-cyan-300'
                        : 'bg-slate-900 border-slate-800 text-slate-500'
                    }`}
                  >
                    {welcomeVoiceEnabled ? 'BẬT CHÀO' : 'TẮT'}
                  </button>
                </div>

                <p className="text-[10px] text-slate-400 italic">
                  💡 Người dùng chỉ cần đứng trước camera {touchlessDelay}s, AI sẽ tự động quét và nhả giấy mà không cần chạm tay vào màn hình.
                </p>
              </div>
            ) : (
              <p className="text-[11px] text-slate-400">
                Ở chế độ bấm nút tay, người dùng cần chạm vào nút lớn <strong>"NHẬN GIẤY VỆ SINH"</strong> để kích hoạt camera nhả giấy.
              </p>
            )}
          </div>

          <div className="border-t border-slate-800/80" />

          {/* SECTION 7: DEVICE INFO */}
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
