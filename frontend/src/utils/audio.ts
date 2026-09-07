/**
 * Natural Vietnamese Voice Assistant & Sound Synthesizer
 * for Smart Toilet Paper Dispenser.
 *
 * Uses:
 * 1. Native Vietnamese Voice TTS (Google Vietnamese Speech) via Backend Cache / Stream.
 *    Eliminates English accent issues completely.
 * 2. Web Audio API synthesized chimes/tones (pleasant 2-tone chime & alert sound).
 * 3. Graceful offline fallback to Web Speech API.
 */

class AudioController {
  private ctx: AudioContext | null = null;
  private soundEnabled: boolean = true;
  private voiceEnabled: boolean = true;
  private voiceVolume: number = 1.0;
  private voiceRate: number = 1.0;
  private currentAudio: HTMLAudioElement | null = null;
  private cachedVoice: SpeechSynthesisVoice | null = null;

  constructor() {
    if (typeof window !== 'undefined' && 'speechSynthesis' in window) {
      window.speechSynthesis.onvoiceschanged = () => {
        this.findVietnameseVoice();
      };
      this.findVietnameseVoice();
    }
  }

  public setSoundEnabled(enabled: boolean) {
    this.soundEnabled = enabled;
    if (!enabled) {
      this.stopSpeaking();
    }
  }

  public setVoiceConfig(enabled: boolean, volume: number = 1.0, rate: number = 1.0) {
    this.voiceEnabled = enabled;
    this.voiceVolume = Math.max(0.1, Math.min(1.0, volume));
    this.voiceRate = Math.max(0.6, Math.min(1.5, rate));
    if (!enabled) {
      this.stopSpeaking();
    }
  }

  public isVoiceEnabled(): boolean {
    return this.voiceEnabled;
  }

  private findVietnameseVoice(): SpeechSynthesisVoice | null {
    if (typeof window === 'undefined' || !('speechSynthesis' in window)) return null;

    const voices = window.speechSynthesis.getVoices();
    if (!voices || voices.length === 0) return null;

    const viVoice = voices.find(
      (v) =>
        v.lang.toLowerCase().includes('vi') ||
        v.name.toLowerCase().includes('vietnam') ||
        v.name.toLowerCase().includes('hoaimy') ||
        v.name.toLowerCase().includes('namminh')
    );

    if (viVoice) {
      this.cachedVoice = viVoice;
      return viVoice;
    }

    return null;
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
   * Speak a phrase in 100% natural native Vietnamese using the backend TTS engine.
   */
  public speak(text: string) {
    if (!this.voiceEnabled) return;
    if (typeof window === 'undefined') return;

    try {
      this.stopSpeaking();

      // Native Vietnamese TTS Stream (Crystal-clear pronunciation, zero English accent)
      const ttsUrl = `/api/v1/dispenser/tts?text=${encodeURIComponent(text.trim())}`;
      const audio = new Audio(ttsUrl);
      audio.volume = this.voiceVolume;
      audio.playbackRate = this.voiceRate;
      this.currentAudio = audio;

      audio.play().catch(() => {
        // Fallback to browser speechSynthesis if audio element play is blocked or offline
        this.speakFallbackSpeechSynthesis(text);
      });
    } catch {
      this.speakFallbackSpeechSynthesis(text);
    }
  }

  private speakFallbackSpeechSynthesis(text: string) {
    if (typeof window === 'undefined' || !('speechSynthesis' in window)) return;
    try {
      window.speechSynthesis.cancel();
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
      // Ignored
    }
  }

  public stopSpeaking() {
    if (this.currentAudio) {
      try {
        this.currentAudio.pause();
        this.currentAudio.currentTime = 0;
      } catch {}
      this.currentAudio = null;
    }
    if (typeof window !== 'undefined' && 'speechSynthesis' in window) {
      try {
        window.speechSynthesis.cancel();
      } catch {}
    }
  }

  // =========================================================================
  // AUDIO SYNTHESIZER SOUND EFFECTS (Web Audio API)
  // =========================================================================

  public playGranted() {
    if (!this.soundEnabled) return;
    try {
      const ctx = this.getContext();
      if (!ctx) return;

      const now = ctx.currentTime;
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
      // Ignored
    }
  }

  public playBlocked() {
    if (!this.soundEnabled) return;
    try {
      const ctx = this.getContext();
      if (!ctx) return;

      const now = ctx.currentTime;
      const osc = ctx.createOscillator();
      const gain = ctx.createGain();
      osc.type = 'triangle';
      osc.frequency.setValueAtTime(329.63, now);
      osc.frequency.exponentialRampToValueAtTime(220.0, now + 0.4);
      gain.gain.setValueAtTime(0.18, now);
      gain.gain.exponentialRampToValueAtTime(0.001, now + 0.45);
      osc.connect(gain);
      gain.connect(ctx.destination);
      osc.start(now);
      osc.stop(now + 0.45);
    } catch {
      // Ignored
    }
  }

  // =========================================================================
  // SCENARIO-BASED VIETNAMESE VOICE ANNOUNCEMENTS (Chime + Voice Sequence)
  // =========================================================================

  public announceGranted(name?: string | null, isNewUser?: boolean) {
    this.playGranted();
    if (!this.voiceEnabled) return;

    setTimeout(() => {
      if (isNewUser) {
        this.speak("Chào mừng bạn! Hệ thống đang cấp giấy vệ sinh, xin mời nhận giấy.");
      } else if (name && !name.startsWith("USER_")) {
        this.speak(`Xin chào ${name}! Hệ thống đang cấp giấy vệ sinh cho bạn.`);
      } else {
        this.speak("Nhận diện thành công! Hệ thống đang cấp giấy vệ sinh, xin mời nhận giấy.");
      }
    }, 400);
  }

  public announceBlocked(secondsRemaining: number) {
    this.playBlocked();
    if (!this.voiceEnabled) return;

    setTimeout(() => {
      const m = Math.floor(secondsRemaining / 60);
      const s = secondsRemaining % 60;

      if (m > 0) {
        this.speak(`Bạn vừa mới nhận giấy vệ sinh. Vui lòng chờ thêm ${m} phút nữa trước khi lấy lần tiếp theo.`);
      } else {
        this.speak(`Bạn vừa mới nhận giấy vệ sinh. Vui lòng chờ thêm ${s} giây nữa nhé.`);
      }
    }, 400);
  }

  public announceMaskDetected() {
    this.playBlocked();
    if (!this.voiceEnabled) return;
    setTimeout(() => {
      this.speak("Vui lòng tháo khẩu trang để hệ thống nhận diện khuôn mặt.");
    }, 400);
  }

  public announceOcclusion() {
    this.playBlocked();
    if (!this.voiceEnabled) return;
    setTimeout(() => {
      this.speak("Khuôn mặt đang bị che khuất. Vui lòng bỏ tay hoặc vật cản trước mặt.");
    }, 400);
  }

  public announceNoFace() {
    this.playBlocked();
    if (!this.voiceEnabled) return;
    setTimeout(() => {
      this.speak("Không tìm thấy khuôn mặt. Vui lòng đứng đối diện trước camera.");
    }, 400);
  }

  public announceSpoofDetected() {
    this.playBlocked();
    if (!this.voiceEnabled) return;
    setTimeout(() => {
      this.speak("Cảnh báo hình ảnh không hợp lệ. Vui lòng thử lại trực tiếp trước camera.");
    }, 400);
  }

  public testVoice() {
    this.speak("Xin chào! Đây là thông báo giọng nói tiếng Việt của máy cấp giấy vệ sinh thông minh.");
  }
}

export const soundEffects = new AudioController();
