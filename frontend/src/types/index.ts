export interface Person {
  person_id: string;
  name: string;
  department?: string;
  role?: string;
  metadata?: Record<string, any>;
  status: 'active' | 'inactive' | 'suspended';
  created_at?: string;
  updated_at?: string;
}

export interface Candidate {
  person_id: string;
  similarity: number;
  name?: string;
}

export interface RecognitionResult {
  status: 'MATCHED' | 'UNKNOWN' | 'AMBIGUOUS' | 'NO_FACE' | 'LOW_QUALITY' | 'MASK_DETECTED' | 'OCCLUDED_FACE' | string;
  person_id: string | null;
  similarity: number;
  second_similarity?: number;
  margin?: number;
  candidates?: Candidate[];
  model_version?: string;
}

export interface AccessEvent {
  id?: string;
  event_type: 'ACCESS_EVENT' | 'SECURITY_ALERT' | 'DISPENSER_TRIGGER' | 'DISPENSER_BLOCKED' | 'CONNECTED' | 'PONG' | string;
  person_id?: string | null;
  name?: string | null;
  device_id?: string;
  similarity?: number | null;
  margin?: number | null;
  status: 'MATCHED' | 'UNKNOWN' | 'AMBIGUOUS' | 'MASK_DETECTED' | 'OCCLUDED_FACE' | 'SPOOF_DETECTED' | 'GRANTED' | 'NEW_USER_GRANTED' | 'COOLDOWN_BLOCKED' | string;
  liveness?: 'PASS' | 'FAIL' | 'INCONCLUSIVE' | string;
  pulse_ms?: number;
  cooldown_seconds_remaining?: number;
  message?: string;
  timestamp: string;
  metadata?: Record<string, any>;
}

export type ConnectionStatus = 'connected' | 'connecting' | 'disconnected';

export interface DispenseResult {
  granted: boolean;
  status: 'GRANTED' | 'NEW_USER_GRANTED' | 'COOLDOWN_BLOCKED' | 'MASK_DETECTED' | 'OCCLUSION_DETECTED' | 'NO_FACE_DETECTED' | 'SPOOF_DETECTED' | string;
  person_id?: string | null;
  name?: string | null;
  is_new_user?: boolean;
  similarity?: number;
  cooldown_remaining_seconds?: number;
  last_dispensed_at?: string;
  dispensed_at?: string;
  pulse_ms?: number;
  message: string;
}

export interface DispenserStats {
  total_granted: number;
  total_blocked: number;
  unique_users: number;
  last_dispensed_at?: string | null;
}

export interface DispenseLog {
  id: number;
  person_id: string;
  name: string;
  device_id: string;
  status: string;
  similarity: number;
  cooldown_seconds_remaining: number;
  message: string;
  timestamp: string;
}

export interface DispenserConfig {
  cooldown_minutes: number;
  dismiss_seconds: number;
  pulse_ms: number;
  device_id: string;
  device_name: string;
  viewfinder_style?: 'hud' | 'corners' | 'oval';
}

