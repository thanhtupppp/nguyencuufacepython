/**
 * Enhanced Audio Controller & Vietnamese Voice Announcement System
 * for Smart Toilet Paper Dispenser.
 *
 * Provides:
 * 1. Web Audio API synthesized chimes/tones (zero external mp3 assets needed).
 * 2. Web Speech API Vietnamese Voice Announcements (Offline, instant, natural pronunciation).
 */

class AudioController {
  private ctx: AudioContext | null = null;
  private soundEnabled: boolean = true;
  private voiceEnabled: boolean = true;
  private voiceVolume: number = 1.0;
  private voiceRate: number = 1.0;
  private cachedVoice: SpeechSynthesisVoice | null = null;

  constructor() {
    if (typeof window !== 'undefined' && 'speechSynthesis' in window) {
      // Chrome/Edge loads voices asynchronously
      window.speechSynthesis.onvoiceschanged = () => {
        this.findVietnameseVoice();
      };
      this.findVietnameseVoice();
    }
  }

  public setSoundEnabled(enabled: boolean) {
    this.soundEnabled = enabled;
  }

  public setVoiceConfig(enabled: boolean, volume: number = 1.0, rate: number = 1.0) {
    this.voiceEnabled = enabled;
    this.voiceVolume = Math.max(0.1, Math.min(1.0, volume));
    this.voiceRate = Math.max(0.6, Math.min(1.5, rate));
  }

  public isVoiceEnabled(): boolean {
    return this.voiceEnabled;
  }

  private findVietnameseVoice(): SpeechSynthesisVoice | null {
    if (typeof window === 'undefined' || !('speechSynthesis' in window)) return null;

    const voices = window.speechSynthesis.getVoices();
    if (!voices || voices.length === 0) return null;

    // Prioritize native Vietnamese voices (e.g. Google Tiếng Việt, Microsoft HoaiMy, Microsoft NamMinh)
    const viVoice = voices.find(
      (v) =>
        v.lang.toLowerCase() === 'vi-vn' ||
        v.lang.toLowerCase().startsWith('vi') ||
        v.name.toLowerCase().includes('vietnam') ||
        v.name.toLowerCase().includes('vietnamese') ||
        v.name.toLowerCase().includes('hoaimy') ||
        v.name.toLowerCase().includes('namminh')
    );

    if (viVoice) {
      this.cachedVoice = viVoice;
      return viVoice;
    }

    return voices[0] || null;
  }

  private getContext(): AudioContext | null {
    if (typeof window === 'undefined') return null;
    if (!this.ctx) {
      const AudioCtx = window.AudioContext || (window as any).webkitAudioContext;
      if (AudioCtx) {
        this.ctx = new AudioCtx();
      }
    }
    if (this.ctx && this.ctx.state === 'suspended') {
      this.ctx.resume().catch(() => {});
    }
    return this.ctx;
  }

  /**
   * Speak a phrase in Vietnamese using the Web Speech API.
   */
  public speak(text: string) {
    if (!this.voiceEnabled) return;
    if (typeof window === 'undefined' || !('speechSynthesis' in window)) return;

    try {
      window.speechSynthesis.cancel(); // Stop previous voice playback to avoid overlapping

      const utterance = new SpeechSynthesisUtterance(text);
      utterance.lang = 'vi-VN';
      utterance.volume = this.voiceVolume;
      utterance.rate = this.voiceRate;
      utterance.pitch = 1.0;

      const voice = this.cachedVoice || this.findVietnameseVoice();
      if (voice) {
        utterance.voice = voice;
      }

      window.speechSynthesis.speak(utterance);
    } catch {
      // Speech playback failed gracefully
    }
  }

  public stopSpeaking() {
    if (typeof window !== 'undefined' && 'speechSynthesis' in window) {
      window.speechSynthesis.cancel();
    }
  }

  // =========================================================================
  // AUDIO SYNTHESIZER SOUND EFFECTS
  // =========================================================================

  /**
   * Pleasant ascending 2-tone chime for successful dispense.
   */
  public playGranted() {
    if (!this.soundEnabled) return;
    try {
      const ctx = this.getContext();
      if (!ctx) return;

      const now = ctx.currentTime;
      // Tone 1 (C5 - 523.25 Hz)
      const osc1 = ctx.createOscillator();
      const gain1 = ctx.createGain();
      osc1.type = 'sine';
      osc1.frequency.setValueAtTime(523.25, now);
      gain1.gain.setValueAtTime(0.15, now);
      gain1.gain.exponentialRampToValueAtTime(0.001, now + 0.35);
      osc1.connect(gain1);
      gain1.connect(ctx.destination);
      osc1.start(now);
      osc1.stop(now + 0.35);

      // Tone 2 (E5 - 659.25 Hz)
      const osc2 = ctx.createOscillator();
      const gain2 = ctx.createGain();
      osc2.type = 'sine';
      osc2.frequency.setValueAtTime(659.25, now + 0.12);
      gain2.gain.setValueAtTime(0.18, now + 0.12);
      gain2.gain.exponentialRampToValueAtTime(0.001, now + 0.6);
      osc2.connect(gain2);
      gain2.connect(ctx.destination);
      osc2.start(now + 0.12);
      osc2.stop(now + 0.6);
    } catch {
      // Audio playback failed silently
    }
  }

  /**
   * Gentle descending warning tone for cooldown block or alert.
   */
  public playBlocked() {
    if (!this.soundEnabled) return;
    try {
      const ctx = this.getContext();
      if (!ctx) return;

      const now = ctx.currentTime;
      const osc = ctx.createOscillator();
      const gain = ctx.createGain();
      osc.type = 'triangle';
      osc.frequency.setValueAtTime(329.63, now); // E4
      osc.frequency.exponentialRampToValueAtTime(220.0, now + 0.4); // A3
      gain.gain.setValueAtTime(0.18, now);
      gain.gain.exponentialRampToValueAtTime(0.001, now + 0.45);
      osc.connect(gain);
      gain.connect(ctx.destination);
      osc.start(now);
      osc.stop(now + 0.45);
    } catch {
      // Audio playback failed silently
    }
  }

  // =========================================================================
  // SCENARIO-BASED VIETNAMESE VOICE ANNOUNCEMENTS
  // =========================================================================

  /**
   * Voice prompt when toilet paper is dispensed successfully.
   */
  public announceGranted(name?: string | null, isNewUser?: boolean) {
    this.playGranted();
    if (!this.voiceEnabled) return;

    if (isNewUser) {
      this.speak("Chào mừng bạn! Hệ thống đang cấp giấy vệ sinh, xin mời nhận giấy.");
    } else if (name && !name.startsWith("USER_")) {
      this.speak(`Xin chào ${name}! Hệ thống đang cấp giấy vệ sinh cho bạn.`);
    } else {
      this.speak("Nhận diện thành công! Hệ thống đang cấp giấy vệ sinh, xin mời nhận giấy.");
    }
  }

  /**
   * Voice prompt when blocked by cooldown to prevent waste.
   */
  public announceBlocked(secondsRemaining: number) {
    this.playBlocked();
    if (!this.voiceEnabled) return;

    const m = Math.floor(secondsRemaining / 60);
    const s = secondsRemaining % 60;

    if (m > 0) {
      this.speak(`Bạn vừa mới nhận giấy vệ sinh. Vui lòng chờ thêm ${m} phút nữa trước khi lấy lần tiếp theo.`);
    } else {
      this.speak(`Bạn vừa mới nhận giấy vệ sinh. Vui lòng chờ thêm ${s} giây nữa nhé.`);
    }
  }

  /**
   * Voice prompt when a face mask is detected.
   */
  public announceMaskDetected() {
    this.playBlocked();
    if (!this.voiceEnabled) return;
    this.speak("Vui lòng tháo khẩu trang để hệ thống nhận diện khuôn mặt.");
  }

  /**
   * Voice prompt when hand or object occludes the face.
   */
  public announceOcclusion() {
    this.playBlocked();
    if (!this.voiceEnabled) return;
    this.speak("Khuôn mặt đang bị che khuất. Vui lòng bỏ tay hoặc vật cản trước mặt.");
  }

  /**
   * Voice prompt when no face is found in frame.
   */
  public announceNoFace() {
    this.playBlocked();
    if (!this.voiceEnabled) return;
    this.speak("Không tìm thấy khuôn mặt. Vui lòng đứng đối diện trước camera.");
  }

  /**
   * Voice prompt when photo/screen spoof is detected.
   */
  public announceSpoofDetected() {
    this.playBlocked();
    if (!this.voiceEnabled) return;
    this.speak("Cảnh báo hình ảnh không hợp lệ. Vui lòng thử lại trực tiếp trước camera.");
  }

  /**
   * Test voice playback button in settings.
   */
  public testVoice() {
    this.speak("Xin chào! Đây là thông báo giọng nói tiếng Việt của máy cấp giấy vệ sinh thông minh.");
  }
}

export const soundEffects = new AudioController();
