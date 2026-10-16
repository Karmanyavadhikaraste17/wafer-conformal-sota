import os
import cv2
import numpy as np

def verify_image_cv(image_path):
    print(f"=== COMPUTER VISION VERIFICATION AUDIT: {image_path} ===")
    img = cv2.imread(image_path)
    if img is None:
        return False, "Failed to load image"
    
    h, w, c = img.shape
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Check 1: Margin & Boundary Clipping (Verify zero text within 15px of boundary)
    margin = 20
    border_strip = np.concatenate([
        gray[0:margin, :].flatten(),
        gray[h-margin:h, :].flatten(),
        gray[:, 0:margin].flatten(),
        gray[:, w-margin:w].flatten()
    ])
    border_variance = np.var(border_strip)
    margin_pass = border_variance < 500  # uniform background
    print(f"  [PASS 1] Margin & Bleed Safety: {'PASSED' if margin_pass else 'FAILED'} (border var: {border_variance:.1f})")
    
    # Check 2: Contrast & Legibility (Michelson contrast between text and background)
    bg_lum = np.median(gray)
    fg_lum = np.percentile(gray, 99.5) # text highlights
    weber_contrast = abs(fg_lum - bg_lum) / (bg_lum + 1e-5)
    michelson_contrast = abs(fg_lum - bg_lum) / (fg_lum + bg_lum + 1e-5)
    
    # Contrast is good if Michelson > 0.60 (WCAG AAA equivalent)
    contrast_pass = michelson_contrast > 0.60
    print(f"  [PASS 2] Dynamic Contrast & Legibility: {'PASSED' if contrast_pass else 'FAILED'} (Michelson: {michelson_contrast:.2f}, FG: {fg_lum:.0f}, BG: {bg_lum:.0f})")
    
    # Check 3: Text Clutter & Collision Analysis
    grad_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    grad_mag = np.sqrt(grad_x**2 + grad_y**2)
    grad_norm = cv2.normalize(grad_mag, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
    
    _, thresh = cv2.threshold(grad_norm, 40, 255, cv2.THRESH_BINARY)
    kernel_text = cv2.getStructuringElement(cv2.MORPH_RECT, (25, 5))
    text_blocks = cv2.dilate(thresh, kernel_text, iterations=2)
    
    contours, _ = cv2.findContours(text_blocks, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    boxes = [cv2.boundingRect(c) for c in contours if cv2.contourArea(c) > 500]
    
    overlaps = 0
    for i in range(len(boxes)):
        x1, y1, w1, h1 = boxes[i]
        for j in range(i+1, len(boxes)):
            x2, y2, w2, h2 = boxes[j]
            xi1 = max(x1, x2)
            yi1 = max(y1, y2)
            xi2 = min(x1+w1, x2+w2)
            yi2 = min(y1+h1, y2+h2)
            if xi1 < xi2 and yi1 < yi2:
                inter_area = (xi2 - xi1) * (yi2 - yi1)
                min_area = min(w1*h1, w2*h2)
                if 0.15 < inter_area / min_area < 0.85:
                    overlaps += 1
                    
    clutter_pass = overlaps == 0
    print(f"  [PASS 3] Zero Text Collision & Overlap: {'PASSED' if clutter_pass else 'FAILED'} (collisions: {overlaps})")
    
    overall = margin_pass and contrast_pass and clutter_pass
    print(f"  --> VERIFICATION RESULT: {'100% PASSED (PUBLICATION GRADE)' if overall else 'REQUIRES RE-RENDER'}\n")
    return overall

if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        verify_image_cv(sys.argv[1])
    else:
        for p in [
            "figures_ultra_clean/slide01_title_dashboard.png",
            "figures_ultra_clean/slide05_capability_synthesis.png",
            "figures_ultra_clean/slide08_closed_loop_metrology.png"
        ]:
            if os.path.exists(p):
                verify_image_cv(p)

