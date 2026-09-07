import { Person, RecognitionResult, DispenseResult, DispenserStats, DispenseLog, DispenserConfig } from '../types';

const API_BASE = import.meta.env.VITE_API_URL || 'http://localhost:8000';

export async function checkHealth(): Promise<boolean> {
  try {
    const res = await fetch(`${API_BASE}/healthz`, { method: 'GET' });
    return res.ok;
  } catch {
    return false;
  }
}

export async function fetchPersons(): Promise<Person[]> {
  const res = await fetch(`${API_BASE}/api/v1/persons`);
  if (!res.ok) throw new Error('Không thể tải danh sách nhân sự');
  return res.json();
}

export async function createPerson(payload: {
  person_id: string;
  name: string;
  department?: string;
  role?: string;
  metadata?: Record<string, any>;
}): Promise<Person> {
  const res = await fetch(`${API_BASE}/api/v1/persons`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(payload),
  });
  if (!res.ok) {
    const err = await res.json().catch(() => ({ detail: 'Lỗi tạo nhân sự' }));
    throw new Error(err.detail || 'Lỗi tạo nhân sự');
  }
  return res.json();
}

export async function updatePersonStatus(person_id: string, status: 'active' | 'inactive' | 'suspended'): Promise<void> {
  const res = await fetch(`${API_BASE}/api/v1/persons/${person_id}/status`, {
    method: 'PATCH',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ status }),
  });
  if (!res.ok) throw new Error('Không thể cập nhật trạng thái');
}

export async function deletePerson(person_id: string): Promise<void> {
  const res = await fetch(`${API_BASE}/api/v1/persons/${person_id}`, {
    method: 'DELETE',
  });
  if (!res.ok && res.status !== 204) throw new Error('Không thể xóa nhân sự');
}

export async function enrollFace(
  person_id: string,
  imageBlob: Blob,
  qualityScore: number = 1.0
): Promise<{ embedding_id: number; person_id: string; quality_score: number }> {
  const formData = new FormData();
  formData.append('person_id', person_id);
  formData.append('image', imageBlob, 'face.jpg');
  formData.append('quality_score', qualityScore.toString());

  const res = await fetch(`${API_BASE}/api/v1/faces/enroll`, {
    method: 'POST',
    body: formData,
  });

  if (!res.ok) {
    const err = await res.json().catch(() => ({ detail: 'Lỗi đăng ký khuôn mặt' }));
    throw new Error(err.detail || 'Lỗi đăng ký khuôn mặt');
  }
  return res.json();
}

export async function recognizeFace(
  imageBlob: Blob,
  threshold: number = 0.60,
  margin: number = 0.08,
  deviceId: string = 'web_dashboard'
): Promise<RecognitionResult> {
  const formData = new FormData();
  formData.append('image', imageBlob, 'capture.jpg');
  formData.append('threshold', threshold.toString());
  formData.append('margin', margin.toString());
  formData.append('device_id', deviceId);

  const res = await fetch(`${API_BASE}/api/v1/faces/recognize`, {
    method: 'POST',
    body: formData,
  });

  if (!res.ok) {
    const err = await res.json().catch(() => ({ detail: 'Nhận diện thất bại' }));
    throw new Error(err.detail || 'Nhận diện thất bại');
  }
  return res.json();
}

export async function requestToiletPaper(
  imageBlob: Blob,
  cooldownMinutes: number = 5.0,
  deviceId: string = 'dispenser_01'
): Promise<DispenseResult> {
  const formData = new FormData();
  formData.append('image', imageBlob, 'user_face.jpg');
  formData.append('cooldown_minutes', cooldownMinutes.toString());
  formData.append('device_id', deviceId);

  const res = await fetch(`${API_BASE}/api/v1/dispenser/request-paper`, {
    method: 'POST',
    body: formData,
  });

  if (!res.ok) {
    const err = await res.json().catch(() => ({ detail: 'Lỗi khi yêu cầu cấp giấy' }));
    throw new Error(err.detail || 'Lỗi khi yêu cầu cấp giấy');
  }
  return res.json();
}

export async function fetchDispenserStats(deviceId?: string): Promise<DispenserStats> {
  const query = deviceId ? `?device_id=${encodeURIComponent(deviceId)}` : '';
  const res = await fetch(`${API_BASE}/api/v1/dispenser/stats${query}`);
  if (!res.ok) throw new Error('Không thể tải dữ liệu thống kê máy cấp giấy');
  return res.json();
}

export async function fetchDispenserLogs(limit: number = 50): Promise<DispenseLog[]> {
  const res = await fetch(`${API_BASE}/api/v1/dispenser/logs?limit=${limit}`);
  if (!res.ok) throw new Error('Không thể tải nhật ký cấp giấy');
  return res.json();
}

export async function fetchDispenserConfig(): Promise<DispenserConfig> {
  const res = await fetch(`${API_BASE}/api/v1/dispenser/config`);
  if (!res.ok) throw new Error('Không thể tải cấu hình máy cấp giấy');
  return res.json();
}

export async function updateDispenserConfig(config: DispenserConfig): Promise<DispenserConfig> {
  const res = await fetch(`${API_BASE}/api/v1/dispenser/config`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(config),
  });
  if (!res.ok) throw new Error('Không thể lưu cấu hình máy cấp giấy');
  const data = await res.json();
  return data.config;
}


