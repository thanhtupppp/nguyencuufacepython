import React, { useState } from 'react';
import { X, Search, Trash2, CheckCircle, RefreshCw, UserCheck, ShieldAlert } from 'lucide-react';
import type { Person } from '../types';
import { updatePersonStatus, deletePerson } from '../services/api';

interface PersonsDrawerProps {
  isOpen: boolean;
  onClose: () => void;
  persons: Person[];
  onRefresh: () => void;
}

export const PersonsDrawer: React.FC<PersonsDrawerProps> = ({
  isOpen,
  onClose,
  persons,
  onRefresh,
}) => {
  const [searchTerm, setSearchTerm] = useState('');
  const [loadingId, setLoadingId] = useState<string | null>(null);

  if (!isOpen) return null;

  const filtered = persons.filter(
    (p) =>
      p.name.toLowerCase().includes(searchTerm.toLowerCase()) ||
      p.person_id.toLowerCase().includes(searchTerm.toLowerCase()) ||
      (p.department && p.department.toLowerCase().includes(searchTerm.toLowerCase()))
  );

  const handleToggleStatus = async (person: Person) => {
    const newStatus = person.status === 'active' ? 'inactive' : 'active';
    setLoadingId(person.person_id);
    try {
      await updatePersonStatus(person.person_id, newStatus);
      onRefresh();
    } catch (err: any) {
      alert(err?.message || 'Lỗi cập nhật trạng thái');
    } finally {
      setLoadingId(null);
    }
  };

  const handleDelete = async (person: Person) => {
    if (!window.confirm(`Bạn có chắc chắn muốn xóa nhân sự ${person.name} (${person.person_id}) cùng toàn bộ vector khuôn mặt?`)) {
      return;
    }
    setLoadingId(person.person_id);
    try {
      await deletePerson(person.person_id);
      onRefresh();
    } catch (err: any) {
      alert(err?.message || 'Lỗi khi xóa nhân sự');
    } finally {
      setLoadingId(null);
    }
  };

  return (
    <div className="fixed inset-0 z-50 overflow-hidden animate-in fade-in duration-200">
      {/* Backdrop */}
      <div
        className="absolute inset-0 bg-slate-950/70 backdrop-blur-xs transition-opacity"
        onClick={onClose}
      />

      <div className="fixed inset-y-0 right-0 max-w-full flex pl-10">
        <div className="w-screen max-w-md bg-slate-900 border-l border-slate-800 shadow-2xl flex flex-col">
          {/* Header */}
          <div className="px-5 py-4 border-b border-slate-800 flex items-center justify-between">
            <div className="flex items-center space-x-2">
              <div className="p-1.5 rounded-lg bg-blue-950/80 text-blue-400 border border-blue-800/40">
                <UserCheck className="w-5 h-5" />
              </div>
              <div>
                <h3 className="font-bold text-white text-base">Danh Sách Nhân Sự</h3>
                <p className="text-xs text-slate-400">Tổng cộng {persons.length} đối tượng</p>
              </div>
            </div>
            <div className="flex items-center space-x-1">
              <button
                onClick={onRefresh}
                className="p-1.5 rounded-lg text-slate-400 hover:text-white hover:bg-slate-800 transition"
                title="Làm mới danh sách"
              >
                <RefreshCw className="w-4 h-4" />
              </button>
              <button
                onClick={onClose}
                className="p-1.5 rounded-lg text-slate-400 hover:text-white hover:bg-slate-800 transition"
              >
                <X className="w-5 h-5" />
              </button>
            </div>
          </div>

          {/* Search box */}
          <div className="p-4 border-b border-slate-800/80 bg-slate-900/50">
            <div className="relative">
              <Search className="w-4 h-4 text-slate-500 absolute left-3 top-2.5" />
              <input
                type="text"
                value={searchTerm}
                onChange={(e) => setSearchTerm(e.target.value)}
                placeholder="Tìm theo tên, mã NV, phòng ban..."
                className="w-full bg-slate-950 border border-slate-800 rounded-xl pl-9 pr-3 py-2 text-xs text-white placeholder-slate-500 focus:outline-none focus:border-cyan-500 transition"
              />
            </div>
          </div>

          {/* List */}
          <div className="flex-1 overflow-y-auto p-4 space-y-2.5">
            {filtered.length === 0 ? (
              <div className="h-40 flex flex-col items-center justify-center text-slate-500 text-xs">
                <span>Không tìm thấy nhân sự phù hợp</span>
              </div>
            ) : (
              filtered.map((person) => {
                const isLoading = loadingId === person.person_id;
                const isActive = person.status === 'active';

                return (
                  <div
                    key={person.person_id}
                    className="p-3.5 rounded-xl border border-slate-800 bg-slate-950/60 hover:border-slate-700 transition flex items-center justify-between gap-3"
                  >
                    <div>
                      <div className="flex items-center space-x-2">
                        <span className="text-sm font-semibold text-white">{person.name}</span>
                        <span className="text-[10px] font-mono px-1.5 py-0.2 bg-slate-800 text-cyan-300 rounded">
                          {person.person_id}
                        </span>
                      </div>
                      <div className="text-xs text-slate-400 mt-1 flex items-center space-x-2">
                        <span>{person.department || 'Chưa phân phòng ban'}</span>
                        {person.role && <span>• {person.role}</span>}
                      </div>
                    </div>

                    {/* Actions */}
                    <div className="flex items-center space-x-1.5">
                      {/* Status toggle */}
                      <button
                        disabled={isLoading}
                        onClick={() => handleToggleStatus(person)}
                        className={`p-1.5 rounded-lg border text-xs transition ${
                          isActive
                            ? 'bg-emerald-950/60 border-emerald-800 text-emerald-300 hover:bg-emerald-900/60'
                            : 'bg-amber-950/60 border-amber-800 text-amber-300 hover:bg-amber-900/60'
                        }`}
                        title={isActive ? 'Nhấp để tạm ngưng hoạt động' : 'Nhấp để kích hoạt'}
                      >
                        {isActive ? (
                          <CheckCircle className="w-4 h-4 text-emerald-400" />
                        ) : (
                          <ShieldAlert className="w-4 h-4 text-amber-400" />
                        )}
                      </button>

                      {/* Delete */}
                      <button
                        disabled={isLoading}
                        onClick={() => handleDelete(person)}
                        className="p-1.5 rounded-lg border border-slate-800 text-slate-400 hover:text-rose-400 hover:border-rose-900/60 hover:bg-rose-950/40 transition"
                        title="Xóa nhân sự"
                      >
                        <Trash2 className="w-4 h-4" />
                      </button>
                    </div>
                  </div>
                );
              })
            )}
          </div>
        </div>
      </div>
    </div>
  );
};
