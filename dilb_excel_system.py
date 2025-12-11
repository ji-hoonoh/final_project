import cv2
import face_recognition
import numpy as np
import os

def draw_lock_screen(width, height):
    """검은색 보안 화면"""
    screen = np.full((height, width, 3), (0, 0, 0), dtype=np.uint8)
    cv2.putText(screen, "SECURITY MODE", (width//2 - 180, height // 2 - 20), 
                cv2.FONT_HERSHEY_DUPLEX, 1.2, (0, 0, 255), 2)
    cv2.putText(screen, "Look at camera to unlock", (width//2 - 150, height // 2 + 40), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (150, 150, 150), 1)
    return screen

def main():
    # 1. 설정
    AUTH_IMAGE_FILE = "me_camera.jpg"
    SECRET_FILE = "secret.xlsx"
    
    # === [여기를 조절하면 더 잘 됩니다] ===
    TOLERANCE = 0.50          # 기준 점수 (0.5 이하면 주인)
    PATIENCE_LIMIT = 20       # 인내심 (20번 실패까지 봐줌)
    CHECK_INTERVAL = 3        # 3프레임마다 검사

    # 파일 확인
    if not os.path.exists(AUTH_IMAGE_FILE):
        print(f"[오류] {AUTH_IMAGE_FILE} 파일이 없습니다.")
        return

    try:
        image = face_recognition.load_image_file(AUTH_IMAGE_FILE)
        authorized_encoding = face_recognition.face_encodings(image)[0]
        print("[시스템] 얼굴 학습 완료. (기준점수: 0.19면 아주 훌륭함)")
    except IndexError:
        print("[오류] 사진에서 얼굴을 못 찾았습니다.")
        return

    # 엑셀 실행
    if os.path.exists(SECRET_FILE):
        os.startfile(SECRET_FILE)

    cap = cv2.VideoCapture(0)
    WINDOW_NAME = "Smart Security Curtain"
    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)

    is_locked = True
    frame_count = 0
    fail_count = 0 
    
    print("=== [실시간 진단 로그 시작] ===")

    while True:
        ret, frame = cap.read()
        if not ret: break
        h, w, _ = frame.shape
        frame_count += 1

        # 얼굴 인식 로직 (주기적으로 실행)
        if frame_count % CHECK_INTERVAL == 0:
            small = cv2.resize(frame, (0,0), fx=0.25, fy=0.25)
            
            # ★★★ [수정된 부분] 여기가 핵심입니다! ★★★
            # 기존: rgb = small[:,:,::-1] (메모리 꼬임 발생)
            # 변경: OpenCV 함수를 써서 깔끔한 RGB 이미지를 만듭니다.
            rgb = cv2.cvtColor(small, cv2.COLOR_BGR2RGB)
            
            face_locs = face_recognition.face_locations(rgb)
            face_encs = face_recognition.face_encodings(rgb, face_locs)
            
            is_owner_detected = False
            current_distance = 1.0 

            # 얼굴이 발견되었을 때만 검사
            if len(face_locs) > 0:
                for enc in face_encs:
                    dist = face_recognition.face_distance([authorized_encoding], enc)[0]
                    current_distance = dist
                    
                    if dist < TOLERANCE:
                        is_owner_detected = True
            
            # === [상태 결정 및 로그 출력] ===
            if is_owner_detected:
                fail_count = 0 
                print(f"\r[O] 주인님 확인됨! (오차: {current_distance:.2f}) - 잠금 해제 유지 중   ", end="")
                
                if is_locked: 
                    cv2.setWindowProperty(WINDOW_NAME, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_NORMAL)
                    cv2.resizeWindow(WINDOW_NAME, 320, 240)
                    cv2.moveWindow(WINDOW_NAME, 0, 0)
                    is_locked = False

            else:
                fail_count += 1
                status_msg = "얼굴 없음" if len(face_locs) == 0 else f"외부인?({current_distance:.2f})"
                print(f"\r[X] 인식 실패 ({fail_count}/{PATIENCE_LIMIT}) - 사유: {status_msg}        ", end="")

                if fail_count > PATIENCE_LIMIT:
                    if not is_locked:
                        print("\n[!!!] 보안 잠금 발동! (엑셀 가림)")
                        cv2.setWindowProperty(WINDOW_NAME, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
                        cv2.setWindowProperty(WINDOW_NAME, cv2.WND_PROP_TOPMOST, 1)
                        is_locked = True

        # 화면 그리기
        if not is_locked:
            display = cv2.resize(frame, (320, 240))
            cv2.rectangle(display, (0,0), (320,240), (0,255,0), 5)
        else:
            display = draw_lock_screen(w, h)
            small_cam = cv2.resize(frame, (0,0), fx=0.3, fy=0.3)
            display[h-small_cam.shape[0]:h, w-small_cam.shape[1]:w] = small_cam

        cv2.imshow(WINDOW_NAME, display)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
