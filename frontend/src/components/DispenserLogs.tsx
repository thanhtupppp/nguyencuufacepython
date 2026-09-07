import React, { useState } from 'react';
import {
  ShieldCheck,
  UserPlus,
  Clock,
  AlertTriangle,
  Radio,
  Sparkles,
  Trash2,
  RefreshCw,
  X,
  CheckCircle2,
  CheckSquare,
  Square,
} from 'lucide-react';
import { DispenseLog, DispenserStats } from '../types';

interface DispenserLogsProps {
  logs: DispenseLog[];
  stats: DispenserStats | null;
  cooldownMinutes?: number;
  onClear: (clearTestUsers: boolean) => Promise<void> | void;
  onRefresh: () => void;
}

export const DispenserLogs: React.FC<DispenserLogsProps> = ({
  logs,
  stats,
  cooldownMinutes = 5,
  onClear,
  onRefresh,
}) => {
  const [showModal, setShowModal] = useState<boolean>(false);
  const [clearTestUsers, setClearTestUsers] = useState<boolean>(true);
  const [isClearing, setIsClearing] = useState<boolean>(false);
  const [toastMessage, setToastMessage] = useState<string | null>(null);

  const handleConfirmClear = async () => {
    try {
      setIsClearing(true);
      await onClear(clearTestUsers);
      setShowModal(false);
      setToastMessage('Đã xóa sạch nhật ký và reset dữ liệu test!');
      setTimeout(() => setToastMessage(null), 3000);
    } finally {
      setIsClearing(false);
    }
  };

  const formatTime = (ts: string) => {
    try {
      const date = new Date(ts);
      return date.toLocaleTimeString('vi-VN', {
        hour: '2-digit',
        minute: '2-digit',
        second: '2-digit',
      });
    } catch {
      return ts;
    }
  };

  const renderStatus = (status: string, cooldownSec: number) => {
    switch (status) {
      case 'GRANTED':
        return (
          <span className="px-2 py-0.5 rounded text-[10px] font-semibold bg-emerald-950/80 border border-emerald-700/60 text-emerald-300 flex items-center space-x-1">
            <ShieldCheck className="w-3 h-3 text-emerald-400" />
            <span>Đã cấp giấy</span>
          </span>
        );
      case 'NEW_USER_GRANTED':
        return (
          <span className="px-2 py-0.5 rounded text-[10px] font-semibold bg-cyan-950/80 border border-cyan-700/60 text-cyan-300 flex items-center space-x-1">
            <UserPlus className="w-3 h-3 text-cyan-400" />
            <span>Người mới cấp giấy</span>
          </span>
        );
      case 'COOLDOWN_BLOCKED':
        return (
          <span className="px-2 py-0.5 rounded text-[10px] font-semibold bg-amber-950/80 border border-amber-700/60 text-amber-300 flex items-center space-x-1">
            <Clock className="w-3 h-3 text-amber-400" />
            <span>Chặn lạm dụng ({cooldownSec > 0 ? `${Math.ceil(cooldownSec / 60)}p` : `${cooldownMinutes}p`})</span>
          </span>
        );
      case 'MASK_DETECTED':
        return (
          <span className="px-2 py-0.5 rounded text-[10px] font-semibold bg-rose-950/80 border border-rose-700/60 text-rose-300 flex items-center space-x-1">
            <AlertTriangle className="w-3 h-3 text-rose-400" />
            <span>Khẩu trang</span>
          </span>
        );
      case 'OCCLUSION_DETECTED':
        return (
          <span className="px-2 py-0.5 rounded text-[10px] font-semibold bg-rose-950/80 border border-rose-700/60 text-rose-300 flex items-center space-x-1">
            <AlertTriangle className="w-3 h-3 text-rose-400" />
            <span>Che mặt</span>
          </span>
        );
      default:
        return (
          <span className="px-2 py-0.5 rounded text-[10px] font-medium bg-slate-800 text-slate-300">
            {status}
          </span>
        );
    }
  };

  return (
    <div className="bg-slate-900 border border-slate-800 rounded-2xl p-4 shadow-xl flex flex-col h-full">
      {/* Metrics Banner */}
      {stats && (
        <div className="grid grid-cols-3 gap-2 pb-4 mb-3 border-b border-slate-800">
          <div className="bg-slate-950/70 p-3 rounded-xl border border-slate-800 text-center">
            <span className="text-[10px] text-slate-400 font-medium block">ĐÃ CẤP GIẤY</span>
            <span className="text-xl font-bold font-mono text-emerald-400">
              {stats.total_granted}
            </span>
          </div>
          <div className="bg-slate-950/70 p-3 rounded-xl border border-slate-800 text-center">
            <span className="text-[10px] text-slate-400 font-medium block">
              CHẶN LẠM DỤNG ({cooldownMinutes}P)
            </span>
            <span className="text-xl font-bold font-mono text-amber-400">
              {stats.total_blocked}
            </span>
          </div>
          <div className="bg-slate-950/70 p-3 rounded-xl border border-slate-800 text-center">
            <span className="text-[10px] text-slate-400 font-medium block">NGƯỜI DÙNG</span>
            <span className="text-xl font-bold font-mono text-cyan-400">
              {stats.unique_users}
            </span>
          </div>
        </div>
      )}

      {/* Toast Notification */}
      {toastMessage && (
        <div className="mb-3 px-3.5 py-2 rounded-xl bg-emerald-950/90 border border-emerald-700/80 text-emerald-300 text-xs flex items-center space-x-2 animate-in fade-in slide-in-from-top-2 duration-200">
          <CheckCircle2 className="w-4 h-4 text-emerald-400 shrink-0" />
          <span className="font-medium">{toastMessage}</span>
        </div>
      )}

      {/* Header */}
      <div className="flex items-center justify-between pb-2 border-b border-slate-800/80">
        <div className="flex items-center space-x-2">
          <div className="p-1.5 rounded-lg bg-cyan-950/80 border border-cyan-800/40 text-cyan-400">
            <Radio className="w-4 h-4" />
          </div>
          <div>
            <h3 className="text-xs sm:text-sm font-semibold text-slate-200 uppercase tracking-wide">
              Nhật ký cấp giấy thời gian thực
            </h3>
            <p className="text-[10px] text-slate-400">Theo dõi sự kiện qua WebSocket</p>
          </div>
        </div>

        {/* Action Buttons */}
        <div className="flex items-center space-x-1.5">
          <button
            onClick={onRefresh}
            className="p-1.5 rounded-lg bg-slate-950 hover:bg-slate-800 border border-slate-800 text-slate-400 hover:text-slate-200 transition"
            title="Làm mới dữ liệu nhật ký"
          >
            <RefreshCw className="w-3.5 h-3.5" />
          </button>

          <button
            onClick={() => setShowModal(true)}
            className="flex items-center space-x-1.5 px-2.5 py-1.5 rounded-xl bg-rose-950/60 hover:bg-rose-900/80 border border-rose-800/80 hover:border-rose-600 text-rose-300 hover:text-white text-xs font-semibold shadow-md shadow-rose-950/30 transition transform active:scale-95"
            title="Xóa toàn bộ nhật ký và reset dữ liệu để test lại từ đầu"
          >
            <Trash2 className="w-3.5 h-3.5 text-rose-400" />
            <span>Xóa Log Test</span>
          </button>
        </div>
      </div>

      {/* Confirmation Modal */}
      {showModal && (
        <div className="fixed inset-0 bg-slate-950/80 backdrop-blur-sm z-50 flex items-center justify-center p-4">
          <div className="bg-slate-900 border border-slate-800 rounded-2xl max-w-md w-full p-5 shadow-2xl space-y-4 animate-in zoom-in-95 duration-150">
            <div className="flex items-center justify-between pb-3 border-b border-slate-800">
              <div className="flex items-center space-x-2 text-rose-400">
                <AlertTriangle className="w-5 h-5" />
                <h3 className="font-bold text-base text-white">Xóa Nhật Ký & Reset Test</h3>
              </div>
              <button
                onClick={() => setShowModal(false)}
                className="p-1 rounded-lg text-slate-400 hover:text-white hover:bg-slate-800 transition"
              >
                <X className="w-4 h-4" />
              </button>
            </div>

            <p className="text-xs text-slate-300 leading-relaxed">
              Bạn có chắc muốn xóa toàn bộ lịch sử cấp giấy trong cơ sở dữ liệu không?
              Hành động này sẽ đặt lại các chỉ số thống kê (đã cấp, chặn lạm dụng) về <strong>0</strong>.
            </p>

            {/* Checkbox option to also remove test guest accounts */}
            <div
              onClick={() => setClearTestUsers(!clearTestUsers)}
              className="flex items-start space-x-2.5 p-3 rounded-xl bg-slate-950 border border-slate-800/80 cursor-pointer hover:border-slate-700 transition"
            >
              <div className="mt-0.5 text-cyan-400">
                {clearTestUsers ? <CheckSquare className="w-4 h-4" /> : <Square className="w-4 h-4 text-slate-500" />}
              </div>
              <div>
                <span className="text-xs font-semibold text-slate-200 block">
                  Xóa tài khoản khách thử nghiệm (USER_*)
                </span>
                <span className="text-[11px] text-slate-400 leading-normal block mt-0.5">
                  Giúp bạn test nhận diện lại khuôn mặt như một <strong>người dùng mới hoàn toàn</strong> (không bị chặn Cooldown 5 phút).
                </span>
              </div>
            </div>

            {/* Modal Actions */}
            <div className="flex items-center justify-end space-x-2 pt-2 border-t border-slate-800">
              <button
                disabled={isClearing}
                onClick={() => setShowModal(false)}
                className="px-4 py-2 rounded-xl text-xs font-medium text-slate-300 hover:text-white hover:bg-slate-800 border border-slate-700 transition"
              >
                Hủy bỏ
              </button>
              <button
                disabled={isClearing}
                onClick={handleConfirmClear}
                className="px-4 py-2 rounded-xl text-xs font-semibold bg-rose-600 hover:bg-rose-500 text-white shadow-lg shadow-rose-600/30 transition flex items-center space-x-1.5 active:scale-95 disabled:opacity-50"
              >
                {isClearing ? (
                  <>
                    <RefreshCw className="w-3.5 h-3.5 animate-spin" />
                    <span>Đang xóa...</span>
                  </>
                ) : (
                  <>
                    <Trash2 className="w-3.5 h-3.5" />
                    <span>Xác Nhận Xóa Sạch</span>
                  </>
                )}
              </button>
            </div>
          </div>
        </div>
      )}

      {/* Scrollable Events List */}
      <div className="flex-1 overflow-y-auto space-y-2 mt-3 pr-1 max-h-[460px]">
        {logs.length === 0 ? (
          <div className="h-40 flex flex-col items-center justify-center text-slate-500 text-xs">
            <Sparkles className="w-6 h-6 text-slate-600 mb-1" />
            <span>Chưa có lượt cấp giấy nào</span>
            <span className="text-[11px] text-slate-600 mt-0.5">
              Bấm nút &quot;Nhận giấy vệ sinh&quot; để ghi nhận lượt đầu tiên
            </span>
          </div>
        ) : (
          logs.map((log, idx) => {
            const isGranted = log.status === 'GRANTED' || log.status === 'NEW_USER_GRANTED';
            return (
              <div
                key={log.id || idx}
                className={`p-3 rounded-xl border transition-all ${
                  isGranted
                    ? 'bg-slate-950/60 border-slate-800/80 hover:border-emerald-700/50'
                    : 'bg-amber-950/20 border-amber-900/30 hover:border-amber-700/60'
                }`}
              >
                <div className="flex items-start justify-between gap-2">
                  <div>
                    <div className="flex items-center space-x-2">
                      <span className="text-xs font-bold text-white">
                        {log.name || log.person_id}
                      </span>
                      <span className="text-[10px] font-mono text-slate-400 bg-slate-800/80 px-1.5 py-0.2 rounded">
                        {log.person_id}
                      </span>
                    </div>
                    <div className="text-[11px] text-slate-400 mt-1 line-clamp-1">
                      {log.message || (isGranted ? 'Cấp giấy thành công' : 'Từ chối cấp giấy')}
                    </div>
                  </div>

                  <div className="flex flex-col items-end space-y-1 shrink-0">
                    {renderStatus(log.status, log.cooldown_seconds_remaining)}
                    <span className="text-[10px] font-mono text-slate-500">
                      {formatTime(log.timestamp)}
                    </span>
                  </div>
                </div>
              </div>
            );
          })
        )}
      </div>
    </div>
  );
};
