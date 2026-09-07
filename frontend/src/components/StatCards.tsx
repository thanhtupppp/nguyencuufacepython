import React from 'react';
import { Users, Scan, UserCheck, AlertTriangle } from 'lucide-react';

interface StatCardsProps {
  totalPersons: number;
  totalScans: number;
  matchedScans: number;
  alertScans: number;
}

export const StatCards: React.FC<StatCardsProps> = ({
  totalPersons,
  totalScans,
  matchedScans,
  alertScans,
}) => {
  const cards = [
    {
      title: 'Nhân sự đã đăng ký',
      value: totalPersons,
      icon: Users,
      color: 'text-blue-400',
      bgColor: 'bg-blue-950/40',
      borderColor: 'border-blue-900/40',
      subText: 'Dữ liệu SQLite Vector DB',
    },
    {
      title: 'Lượt quét phiên này',
      value: totalScans,
      icon: Scan,
      color: 'text-cyan-400',
      bgColor: 'bg-cyan-950/40',
      borderColor: 'border-cyan-900/40',
      subText: 'Camera AI Real-time',
    },
    {
      title: 'Xác thực thành công',
      value: matchedScans,
      icon: UserCheck,
      color: 'text-emerald-400',
      bgColor: 'bg-emerald-950/40',
      borderColor: 'border-emerald-900/40',
      subText: totalScans > 0 ? `${Math.round((matchedScans / totalScans) * 100)}% tỷ lệ nhận diện` : 'Chưa có lượt quét',
    },
    {
      title: 'Cảnh báo / Lạ mặt',
      value: alertScans,
      icon: AlertTriangle,
      color: 'text-amber-400',
      bgColor: 'bg-amber-950/40',
      borderColor: 'border-amber-900/40',
      subText: 'Unknown, Mask hoặc Spoof',
    },
  ];

  return (
    <div className="grid grid-cols-2 lg:grid-cols-4 gap-3.5 mb-6">
      {cards.map((card, idx) => {
        const Icon = card.icon;
        return (
          <div
            key={idx}
            className={`p-4 rounded-xl border ${card.borderColor} ${card.bgColor} backdrop-blur-sm transition-all hover:translate-y-[-2px] shadow-sm`}
          >
            <div className="flex items-center justify-between">
              <span className="text-xs font-medium text-slate-400">{card.title}</span>
              <div className={`p-2 rounded-lg bg-slate-900/60 ${card.color}`}>
                <Icon className="w-4 h-4" />
              </div>
            </div>
            <div className="mt-2 flex items-baseline space-x-2">
              <span className="text-2xl font-bold font-mono tracking-tight text-white">
                {card.value}
              </span>
            </div>
            <div className="mt-1 text-[11px] text-slate-500 truncate">{card.subText}</div>
          </div>
        );
      })}
    </div>
  );
};
