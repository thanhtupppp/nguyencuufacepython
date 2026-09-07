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
  event_type: 'ACCESS_EVENT' | 'SECURITY_ALERT' | 'CONNECTED' | 'PONG' | string;
  person_id?: string | null;
  name?: string | null;
  device_id?: string;
  similarity?: number | null;
  margin?: number | null;
  status: 'MATCHED' | 'UNKNOWN' | 'AMBIGUOUS' | 'MASK_DETECTED' | 'OCCLUDED_FACE' | 'SPOOF_DETECTED' | string;
  liveness?: 'PASS' | 'FAIL' | 'INCONCLUSIVE' | string;
  timestamp: string;
  metadata?: Record<string, any>;
}

export type ConnectionStatus = 'connected' | 'connecting' | 'disconnected';
