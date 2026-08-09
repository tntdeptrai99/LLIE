import re
from pathlib import Path

def main():
    script_path = Path("scripts/create_miwai_reportV1.py")
    with open(script_path, "r", encoding="utf-8") as f:
        content = f.read()

    # 1. Update Headings
    content = content.replace("doc.add_heading('4.2 Đánh giá Hàm Suy hao', level=2)", 
                              "doc.add_heading('4.2 Ablation Thành phần (Component Ablation)', level=2)")
    
    content = content.replace("doc.add_heading('4.3 Kiến trúc Mô hình', level=2)", 
                              "doc.add_heading('4.3 Benchmark Kiến trúc Mô hình (Architecture Benchmark)', level=2)")
    
    content = content.replace("add_paragraph(doc, \"Đánh giá các phương pháp tăng cường hình ảnh: Bảng 1", 
                              "doc.add_heading('4.4 So sánh Định lượng (Quantitative Comparison)', level=2)\n    add_paragraph(doc, \"Đánh giá các phương pháp tăng cường hình ảnh: Bảng 1")
    
    content = content.replace("doc.add_heading('4.3 Đánh giá Thị giác Thực tế', level=2)", 
                              "doc.add_heading('4.5 Đánh giá Thị giác Thực tế', level=2)")
    
    content = content.replace("doc.add_heading('4.4 Đánh giá Triển khai trên Vi điều khiển (STM32H750)', level=2)", 
                              "doc.add_heading('4.6 Đánh giá Triển khai trên Vi điều khiển (STM32H750)', level=2)")
    
    # 2. Insert Board Benchmark Section
    board_section = """
    doc.add_heading('4.7 Benchmark Thị giác Thực tế trên Board', level=2)
    add_paragraph(doc, "Ngoài việc đo lường thời gian suy luận và dung lượng bộ nhớ, chúng tôi đã trích xuất trực tiếp kết quả xử lý ảnh (tensor dump) từ bo mạch STM32H750 để đánh giá chất lượng thị giác đầu ra thực tế của mô hình lượng tử hóa DG-GhostESP-96 so với ảnh đầu vào.")
    
    add_paragraph(doc, "Bảng 3: So sánh các chỉ số độ sáng, độ tương phản và độ nét giữa ảnh đầu vào (input_preprocess) và ảnh đầu ra trên board (ai_output).")
    board_headers = ["Ảnh", "Brightness", "Contrast", "Saturation", "Sharpness", "Clip 0", "Clip 255"]
    board_data = [
        ["input_preprocess", "38.8530", "31.2182", "0.5018", "13.8072", "0.119936", "0.000000"],
        ["ai_output", "101.9518", "48.1441", "0.2610", "22.0063", "0.000000", "0.003906"]
    ]
    add_table(doc, board_headers, board_data)
    
    add_paragraph(doc, "Kết quả từ Bảng 3 cho thấy ảnh đầu ra từ AI có độ sáng trung bình (Brightness) và độ tương phản (Contrast) tăng vọt, trong khi độ nét (Sharpness) cũng được cải thiện đáng kể. Đồng thời, tỷ lệ pixel bị cắt tối (Clip 0) giảm từ ~12% xuống 0%, chứng tỏ khả năng khôi phục chi tiết vùng tối rất tốt của mô hình lượng tử hóa.")
    
    add_figure(doc, FIG_DIR / "fig10_board_visual.png", "Hình 10: So sánh thị giác thực tế trên board: input preprocess và AI output.")
    
    add_figure(doc, FIG_DIR / "fig11_board_histogram.png", "Hình 11: Histogram độ sáng trước/sau trên frame board thực tế. (Đường màu xanh: input, Đường màu đỏ: AI output).")
    
    doc.add_heading('5. Kết luận và Hướng phát triển', level=1)"""
    
    content = content.replace("doc.add_heading('5. Kết luận và Hướng phát triển', level=1)", board_section)
    
    # 3. Fix Figure Numbering globally using regex or replace
    content = content.replace("Hình 3: Tiến trình tối ưu", "Hình 3: Tiến trình tối ưu")
    content = content.replace("Hình 4: So sánh hiệu suất định lượng", "Hình 4: So sánh hiệu suất định lượng")
    content = content.replace("Hình 3: So sánh đường cong", "Hình 5: So sánh đường cong")
    content = content.replace("Hình 4: So sánh hiệu suất huấn luyện", "Hình 6: So sánh hiệu suất huấn luyện")
    
    # "Hình 5: Benchmark thị giác thực tế." -> Hình 7
    content = content.replace("Hình 5: Benchmark thị giác thực tế.", "Hình 7: Benchmark thị giác thực tế.")
    content = content.replace("(Hình 5)", "(Hình 7)")
    
    # "Hình 6: Sơ đồ hệ thống" -> Hình 8
    content = content.replace("Hình 6: Sơ đồ hệ thống tổng thể", "Hình 8: Sơ đồ hệ thống tổng thể")
    
    # "Hình 7: Biểu đồ hộp" -> Hình 9
    content = content.replace("Hình 7: Biểu đồ hộp", "Hình 9: Biểu đồ hộp")
    content = content.replace("Hình 6, cấu trúc lượng", "Hình 9, cấu trúc lượng")
    
    with open(script_path, "w", encoding="utf-8") as f:
        f.write(content)
        
    print("Successfully reformatted script.")

if __name__ == "__main__":
    main()
