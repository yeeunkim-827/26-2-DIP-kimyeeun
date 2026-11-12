import io
import numpy as np
import streamlit as st
from PIL import Image, ImageDraw
from scipy import ndimage as ndi

st.set_page_config(page_title='Object Counting', page_icon='🔎', layout='wide')
st.title('🔎 Object Counting Web App')
st.write('이미지를 업로드하면 영상처리를 통해 객체를 검출하고 개수를 계산합니다.')

with st.sidebar:
    st.header('⚙️ 분석 설정')
    blur = st.slider('Median filter 크기', 1, 11, 3, step=2)
    threshold = st.slider('이진화 임계값', 0, 255, 125)
    polarity = st.radio('객체 색상', ['밝은 객체 (어두운 배경)', '어두운 객체 (밝은 배경)'])
    min_size = st.number_input('최소 객체 면적 (pixel)', min_value=1, max_value=1000000, value=100, step=50)
    max_size = st.number_input('최대 객체 면적 (pixel)', min_value=1, max_value=100000000, value=10000000, step=1000)
    fill_holes = st.checkbox('객체 내부 구멍 채우기', value=True)

uploaded = st.file_uploader('이미지 파일을 선택하세요', type=['png', 'jpg', 'jpeg'])
if uploaded is None:
    st.info('분석할 이미지를 업로드해 주세요.')
    st.stop()

try:
    image = Image.open(uploaded).convert('RGB')
except Exception:
    st.error('이미지를 읽을 수 없습니다. 다른 이미지 파일을 선택해 주세요.')
    st.stop()

# 큰 사진에서도 웹 앱이 무리 없이 작동하도록 크기 제한
image.thumbnail((1600, 1600))
gray = np.asarray(image.convert('L'))
filtered = ndi.median_filter(gray, size=blur)
if polarity.startswith('밝은'):
    mask = filtered > threshold
else:
    mask = filtered < threshold
if fill_holes:
    mask = ndi.binary_fill_holes(mask)

# 8-연결성 기준 객체 분리 및 크기 필터링
labels, _ = ndi.label(mask, structure=np.ones((3, 3), dtype=np.uint8))
sizes = np.bincount(labels.ravel())
valid = (sizes >= min_size) & (sizes <= max_size)
valid[0] = False
kept = valid[labels]
final_labels, count = ndi.label(kept, structure=np.ones((3, 3), dtype=np.uint8))

result = image.copy()
draw = ImageDraw.Draw(result)
objects = ndi.find_objects(final_labels)
for idx, slc in enumerate(objects, start=1):
    if slc is None:
        continue
    ys, xs = slc
    x0, x1 = xs.start, xs.stop - 1
    y0, y1 = ys.start, ys.stop - 1
    draw.rectangle((x0, y0, x1, y1), outline='#ff3030', width=3)
    draw.text((x0, max(0, y0 - 14)), str(idx), fill='#ff3030', stroke_width=1, stroke_fill='white')

st.metric('검출된 객체 수', int(count))
a, b = st.columns(2)
with a:
    st.subheader('1. 원본 이미지')
    st.image(image, use_container_width=True)
    st.subheader('3. 이진화 및 필터링')
    st.image(Image.fromarray((kept * 255).astype(np.uint8)), use_container_width=True)
with b:
    st.subheader('2. Grayscale + Median Filter')
    st.image(Image.fromarray(filtered), use_container_width=True, clamp=True)
    st.subheader('4. 객체 검출 결과')
    st.image(result, use_container_width=True)

buffer = io.BytesIO()
result.save(buffer, format='PNG')
st.download_button('📥 결과 이미지 다운로드', data=buffer.getvalue(), file_name='object_counting_result.png', mime='image/png')
st.caption('주의: 연결된 객체는 하나로 계산됩니다. 임계값과 최소 면적을 조절하여 결과를 확인하세요.')
