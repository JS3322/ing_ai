from typing import List, Tuple, Iterable, Dict, Optional
import math
import numpy as np

def render_wafer_svg(
    cells: Iterable[Dict],
    bin_edges: List[float],
    palette: List[str],
    size: Tuple[int, int]=(1000, 600),
    plot_box: Tuple[int, int, int, int]=(80, 80, 700, 520),
    wafer: Optional[Tuple[float, float, float]] = None,  # (cx, cy, r) in px
    coord: str = "unit",  # "unit" or "pixel"
    show_mesh: bool = True,
    mesh_step_px: int = 20,
    notch_angle_deg: Optional[float] = 300.0,
    title: Optional[str] = None,
    legend: bool = True,
    legend_box: Optional[Tuple[int,int]] = None,
    outfile: str = "wafer.svg",
) -> str:
    assert len(palette) == len(bin_edges)-1
    W,H = size
    L,T,R,B = plot_box
    plot_w, plot_h = R-L, B-T

    if wafer is None:
        cx, cy = (L+R)/2, (T+B)/2
        r = min(plot_w, plot_h) * 0.46
    else:
        cx, cy, r = wafer

    def unit_to_px(pt: Tuple[float,float]) -> Tuple[float,float]:
        x,y = pt
        return (cx + x*r, cy - y*r)  # SVG y축 보정

    def color_of(v: float) -> str:
        for i in range(len(bin_edges)-1):
            if bin_edges[i] <= v < bin_edges[i+1]:
                return palette[i]
        return palette[-1] if v >= bin_edges[-1] else palette[0]

    parts = []
    parts.append(f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}">')
    parts.append('<style>text{font-family:sans-serif;font-size:12px}</style>')
    if title:
        parts.append(f'<text x="{L}" y="{T-28}" font-size="16">{title}</text>')
    parts.append(f'<rect x="{L}" y="{T}" width="{plot_w}" height="{plot_h}" stroke="black" fill="none" stroke-width="2"/>')

    parts.append('<defs>')
    parts.append('<clipPath id="waferClip">')
    parts.append(f'<circle cx="{cx}" cy="{cy}" r="{r}"/>')
    parts.append('</clipPath>')
    parts.append('</defs>')

    if show_mesh:
        parts.append(f'<g clip-path="url(#waferClip)">')
        for gx in range(int(L), int(R)+1, mesh_step_px):
            parts.append(f'<line x1="{gx}" y1="{T}" x2="{gx}" y2="{B}" stroke="#2b5aa0" stroke-opacity="0.25" stroke-width="1"/>')
        for gy in range(int(T), int(B)+1, mesh_step_px):
            parts.append(f'<line x1="{L}" y1="{gy}" x2="{R}" y2="{gy}" stroke="#2b5aa0" stroke-opacity="0.25" stroke-width="1"/>')
        parts.append('</g>')

    parts.append(f'<g clip-path="url(#waferClip)">')
    for cell in cells:
        poly = cell["poly"]
        v = float(cell["value"])
        pts = [unit_to_px(p) for p in poly] if coord == "unit" else poly
        pts_s = " ".join(f"{x:.2f},{y:.2f}" for x,y in pts)
        parts.append(f'<polygon points="{pts_s}" fill="{color_of(v)}" stroke="none"/>')
    parts.append('</g>')

    parts.append(f'<circle cx="{cx}" cy="{cy}" r="{r}" stroke="navy" fill="none" stroke-width="2"/>')

    if notch_angle_deg is not None:
        ang = math.radians(notch_angle_deg)
        notch_r = r*0.08
        nx = cx + r*math.cos(ang)
        ny = cy - r*math.sin(ang)
        parts.append(f'<path d="M {nx - notch_r:.1f} {ny:.1f} A {notch_r:.1f} {notch_r:.1f} 0 0 1 {nx + notch_r:.1f} {ny:.1f}" stroke="#e74c3c" fill="none" stroke-width="2"/>')
        parts.append(f'<text x="{R+15}" y="{B-40}" fill="#e74c3c">Notch</text>')

    if legend:
        leg_x, leg_y = (R + 40, T + 10) if legend_box is None else legend_box
        sw, sh = 20, 24
        for i in range(len(palette)):
            lo = bin_edges[-(i+2)]
            hi = bin_edges[-(i+1)]
            col = palette[-(i+1)]
            y = leg_y + i*(sh + 4)
            parts.append(f'<rect x="{leg_x}" y="{y}" width="{sw}" height="{sh}" fill="{col}" stroke="black" stroke-width="1"/>')
            parts.append(f'<text x="{leg_x + sw + 8}" y="{y + sh - 8}">{lo:.1f}..{hi:.1f}</text>')

    parts.append('</svg>')
    with open(outfile, "w", encoding="utf-8") as f:
        f.write("\n".join(parts))
    return outfile

def build_square_cells_from_array(arr, grid_extent=(-1.0,-1.0,1.0,1.0), circle_mask=True):
    """2D 배열로부터 사각형 셀들을 생성하는 함수"""
    nx = len(arr[0]); ny = len(arr)
    xmin,ymin,xmax,ymax = grid_extent
    dx = (xmax - xmin) / nx; dy = (ymax - ymin) / ny
    cells = []
    for j in range(ny):
        for i in range(nx):
            x0 = xmin + i*dx; x1 = x0 + dx
            y0 = ymin + j*dy; y1 = y0 + dy
            cx = (x0+x1)/2; cy = (y0+y1)/2
            if (not circle_mask) or (cx*cx + cy*cy <= 1.0):
                poly = [(x0,y0), (x1,y0), (x1,y1), (x0,y1)]
                cells.append({"poly": poly, "value": float(arr[j][i])})
    return cells

def create_sample_wafer_data():
    """샘플 웨이퍼 데이터 생성"""
    # 30x30 그리드 생성
    size = 30
    center = size // 2
    
    # 거리 기반 패턴 생성 (중심에서 멀어질수록 값이 변함)
    data = np.zeros((size, size))
    for i in range(size):
        for j in range(size):
            # 중심에서의 거리 계산
            dist = math.sqrt((i - center)**2 + (j - center)**2)
            # 정규화된 거리 (0~1)
            normalized_dist = dist / (size * 0.5)
            # 패턴 생성 (거리 + 노이즈 + 원형 패턴)
            value = 50 + 30 * math.sin(normalized_dist * math.pi * 2) + np.random.normal(0, 5)
            data[i, j] = value
    
    return data

def main():
    """메인 실행 함수"""
    print("웨이퍼 맵 차트 생성 시작...")
    
    # 샘플 데이터 생성
    wafer_data = create_sample_wafer_data()
    print(f"데이터 크기: {wafer_data.shape}")
    print(f"데이터 범위: {wafer_data.min():.2f} ~ {wafer_data.max():.2f}")
    
    # 셀 데이터 생성
    cells = build_square_cells_from_array(wafer_data, circle_mask=True)
    print(f"생성된 셀 개수: {len(cells)}")
    
    # 색상 범위 설정
    data_min = wafer_data.min()
    data_max = wafer_data.max()
    bin_edges = [
        data_min,
        data_min + (data_max - data_min) * 0.2,
        data_min + (data_max - data_min) * 0.4,
        data_min + (data_max - data_min) * 0.6,
        data_min + (data_max - data_min) * 0.8,
        data_max
    ]
    
    # 색상 팔레트 (낮음->높음: 파란색->빨간색)
    palette = [
        "#2E86AB",  # 진한 파란색
        "#A23B72",  # 보라색
        "#F18F01",  # 주황색
        "#C73E1D",  # 빨간색
        "#8B0000"   # 진한 빨간색 (추가)
    ]
    
    print("SVG 파일 생성 중...")
    
    # SVG 렌더링
    output_file = render_wafer_svg(
        cells=cells,
        bin_edges=bin_edges,
        palette=palette,
        size=(1000, 600),
        plot_box=(80, 80, 700, 520),
        wafer=None,
        coord="unit",
        show_mesh=True,
        mesh_step_px=20,
        notch_angle_deg=None,  # Notch 제거
        title="Sample Wafer Map - 반도체 공정 데이터 시각화",
        legend=True,
        legend_box=None,
        outfile="sample_wafer_map.svg"
    )
    
    print(f"웨이퍼 맵 차트가 생성되었습니다: {output_file}")
    print(f"현재 디렉토리에서 {output_file} 파일을 확인하세요.")

if __name__ == "__main__":
    main()