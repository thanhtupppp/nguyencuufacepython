# THƯ MỤC CHỨA CÁC FILE GHI ÂM GIỌNG NÓI CHO KIOSK CẤP GIẤY
=============================================================================

Bạn hãy chép các file âm thanh (.mp3) sau khi ghi âm vào thư mục này:
`frontend/public/audio/`

Hệ thống sẽ TỰ ĐỘNG ưu tiên phát các file ghi âm thực tế này. Nếu chưa có file nào,
hệ thống sẽ tự động dùng giọng TTS dự phòng.

---

## DANH SÁCH FILE VÀ LỜI THOẠI CẦN GHI ÂM:

1. `grant_new_user.mp3`
   - Kịch bản: "Chào mừng bạn! Hệ thống đang cấp giấy vệ sinh, xin mời nhận giấy."
   - Ngữ điệu: Tươi vui, thân thiện, hiếu khách.

2. `grant_existing.mp3`
   - Kịch bản: "Nhận diện thành công! Hệ thống đang cấp giấy vệ sinh, xin mời nhận giấy."
   - Ngữ điệu: Dứt khoát, lịch sự.

3. `cooldown_blocked.mp3`
   - Kịch bản: "Bạn vừa mới nhận giấy vệ sinh. Vui lòng chờ thêm ít phút trước khi lấy lần tiếp theo!"
   - Ngữ điệu: Nhẹ nhàng, hòa nhã, mang tính nhắc nhở văn minh tránh lãng phí.

4. `mask_alert.mp3`
   - Kịch bản: "Vui lòng tháo khẩu trang để hệ thống nhận diện khuôn mặt."
   - Ngữ điệu: Rõ ràng, dứt khoát.

5. `occlusion_alert.mp3`
   - Kịch bản: "Khuôn mặt đang bị che khuất. Vui lòng bỏ tay hoặc vật cản trước mặt."
   - Ngữ điệu: Rõ ràng, hướng dẫn thao tác.

6. `no_face.mp3`
   - Kịch bản: "Không tìm thấy khuôn mặt. Vui lòng đứng đối diện trước camera."
   - Ngữ điệu: Bình tĩnh, hướng dẫn vị trí đứng.

7. `spoof_alert.mp3`
   - Kịch bản: "Cảnh báo hình ảnh không hợp lệ. Vui lòng thử lại trực tiếp trước camera."
   - Ngữ điệu: Nghiêm túc, dứt khoát.

8. `welcome_guide.mp3`
   - Kịch bản: "Xin chào! Chạm vào nút nhận giấy để bắt đầu nhận diện khuôn mặt."
   - Ngữ điệu: Ấm áp, mời gọi người dùng.

9. `thank_you.mp3` (Tùy chọn)
   - Kịch bản: "Cảm ơn bạn và chúc bạn một ngày tốt lành!"
   - Ngữ điệu: Tươi vui, lịch thiệp.

---

## KHUYẾN NGHỊ KỸ THUẬT KHI GHI ÂM:
- Định dạng file: **.mp3** (hoặc .wav).
- Tần số lấy mẫu (Sample Rate): **44.1 kHz** hoặc **48 kHz**.
- Kênh âm thanh: **Mono** hoặc **Stereo**.
- Tốc độ bit (Bitrate): **128 kbps** hoặc **192 kbps** (đủ rõ nét và dung lượng nhẹ dưới 200KB mỗi file).
- Không gian ghi âm: Nơi yên tĩnh, ít vang (ít echo), âm lượng đồng đều giữa các câu.
