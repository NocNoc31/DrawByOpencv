import cv2
import mediapipe as mp
import numpy as np
import time

# Global variable for the mode
mode = "draw"
show_text = ""  # To store the message that will be displayed on the screen
text_timer = 0  # Timer to control how long the text will be shown

def click_event(x, y):
    global mode
    # Define the button areas (x, y, width, height)
    draw_button = (10, 10, 100, 50)
    erase_button = (120, 10, 100, 50)
    contact_button = (230, 10, 150, 50)

    # Check if the index finger tip is inside the draw button
    if draw_button[0] < x < draw_button[0] + draw_button[2] and draw_button[1] < y < draw_button[1] + draw_button[3]:
        mode = "draw"
    # Check if the index finger tip is inside the erase button
    elif erase_button[0] < x < erase_button[0] + erase_button[2] and erase_button[1] < y < erase_button[1] + erase_button[3]:
        mode = "erase"

def draw_buttons(img):
    # Draw "Draw" button
    cv2.rectangle(img, (10, 10), (110, 60), (0, 255, 0), -1)
    cv2.putText(img, "Draw", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    # Draw "Erase" button
    cv2.rectangle(img, (120, 10), (220, 60), (0, 0, 255), -1)
    cv2.putText(img, "Erase", (130, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

def Detection():
    global mode, show_text, text_timer
    video = cv2.VideoCapture(0)

    mpHands = mp.solutions.hands
    hands = mpHands.Hands(max_num_hands=1, min_detection_confidence=0.7)
    mpDraw = mp.solutions.drawing_utils

    # Create a blank canvas to draw on
    canvas = None

    prev_index_finger_tip = None  # To store previous finger tip position
    draw_color = (0, 0, 255)  # Drawing color (red)
    eraser_size = 50  # Size of eraser

    while True:
        ret, img = video.read()
        if not ret:
            break

        h, w, c = img.shape

        # Initialize canvas if it's not yet created
        if canvas is None:
            canvas = np.zeros_like(img)

        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = hands.process(imgRGB)

        if results.multi_hand_landmarks:
            for handLms in results.multi_hand_landmarks:
                thumb_tip = None
                index_finger_tip = None

                for id, lm in enumerate(handLms.landmark):
                    cx, cy = int(lm.x * w), int(lm.y * h)
                    if id == 4:  # Thumb tip
                        thumb_tip = (cx, cy)
                        cv2.circle(img, (cx, cy), 10, (255, 0, 0), cv2.FILLED)
                    if id == 8:  # Index finger tip
                        index_finger_tip = (cx, cy)
                        cv2.circle(img, (cx, cy), 10, (0, 255, 0), cv2.FILLED)

                if thumb_tip and index_finger_tip:
                    # Distance between thumb and index finger for drawing/erasing
                    distance = np.linalg.norm(np.array(thumb_tip) - np.array(index_finger_tip))
                    
                    if distance < 40:  # Pinch gesture for drawing/erasing
                        if index_finger_tip is not None:
                            # Check if pinch is in button areas
                            click_event(index_finger_tip[0], index_finger_tip[1])

                        if mode == "draw":
                            if prev_index_finger_tip is not None:
                                # Draw a line from the previous position to the current position for smoother drawing
                                cv2.line(canvas, prev_index_finger_tip, index_finger_tip, draw_color, thickness=10)
                            prev_index_finger_tip = index_finger_tip  # Update previous position
                        elif mode == "erase":
                            # Erase by drawing a black circle at the index finger tip position
                            cv2.circle(canvas, index_finger_tip, eraser_size, (0, 0, 0), cv2.FILLED)
                            prev_index_finger_tip = None  # Reset previous position when erasing

                # Draw hand landmarks on the original image
                mpDraw.draw_landmarks(img, handLms, mpHands.HAND_CONNECTIONS)

        # Combine the canvas with the camera feed
        img_with_canvas = cv2.addWeighted(img, 1, canvas, 0.5, 0)

        # Draw the buttons on the image
        draw_buttons(img_with_canvas)

        # Show the current mode on the screen
        cv2.putText(img_with_canvas, f"Mode: {mode.upper()}", (10, 80), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)

        # Show the message if the timer hasn't expired
        if time.time() - text_timer < 2:  # Show text for 2 seconds
            cv2.putText(img_with_canvas, show_text, (10, 200), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2, cv2.LINE_AA)

        cv2.imshow("frame", img_with_canvas)

        k = cv2.waitKey(1)
        if k == ord("q"):  # Press 'q' to quit
            break
        elif k == ord("d"):  # Press 'd' to switch to drawing mode
            mode = "draw"
        elif k == ord("e"):  # Press 'e' to switch to erasing mode
            mode = "erase"
            prev_index_finger_tip = None
        elif k == ord("c"):  # Press 'c' to clear the canvas
            canvas = None
        elif k == ord("s"):  # Press 's' to display the text
            show_text = "Messege has been displayed"
            text_timer = time.time()  # Start the timer for the message

    video.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    Detection()
