import io
import cv2
import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image, ImageOps

st.set_page_config(page_title='Object Counting Lab', page_icon='🔎', layout='wide')
st.title('🔎 Object Counting Lab')
st.caption('HW #6 | Image processing-based object detection and counting')

METHODS = ['자동 추천', '색상 채도 (HSV)', '노란색 객체', '파란색 객체', '밝은 객체', '어두운 객체', '배경과 색상 차이']

def build_mask(rgb, method, threshold, blur):
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    if blur > 1:
        gray = cv2.GaussianBlur(gray, (blur, blur), 0)
    if method == '색상 채도 (HSV)':
        mask = cv2.inRange(hsv, (0, threshold, 35), (179, 255, 255))
    elif method == '노란색 객체':
        mask = cv2.inRange(hsv, (12, max(35, threshold), 45), (43, 255, 255))
    elif method == '파란색 객체':
        mask = cv2.inRange(hsv, (85, max(35, threshold), 30), (140, 255, 255))
    elif method == '밝은 객체':
        _, mask = cv2.threshold(gray, threshold, 255, cv2.THRESH_BINARY)
    elif method == '어두운 객체':
        _, mask = cv2.threshold(gray, threshold, 255, cv2.THRESH_BINARY_INV)
    else:
        # Estimate background from the border, then compare Lab colors.
        lab = cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB).astype(np.float32)
        h, w = lab.shape[:2]
        bw = max(3, min(h, w)//18)
        border = np.concatenate([lab[:bw].reshape(-1,3),lab[-bw:].reshape(-1,3),lab[:, :bw].reshape(-1,3),lab[:, -bw:].reshape(-1,3)])
        bg = np.median(border, axis=0)
        dist = np.linalg.norm(lab-bg, axis=2)
        mask = (dist > max(10, threshold/2)).astype(np.uint8)*255
    return mask

def clean_mask(mask, opening, closing):
    if opening:
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(opening*2+1,opening*2+1))
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN,k)
    if closing:
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(closing*2+1,closing*2+1))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE,k)
    return mask

def components(mask, min_area, max_area, split, peak_ratio):
    if split:
        dist = cv2.distanceTransform(mask,cv2.DIST_L2,5)
        if dist.max() > 0:
            _, peaks = cv2.threshold(dist, peak_ratio*dist.max(),255,cv2.THRESH_BINARY)
            peaks = np.uint8(peaks)
            _, markers = cv2.connectedComponents(peaks)
            markers = markers + 1
            unknown = cv2.subtract(mask,peaks)
            markers[unknown > 0] = 0
            colored = cv2.cvtColor(mask,cv2.COLOR_GRAY2BGR)
            labels = cv2.watershed(colored,markers)
            ids = [int(i) for i in np.unique(labels) if i > 1]
        else:
            labels = np.zeros_like(mask,dtype=np.int32)
            ids=[]
    else:
        _,labels,_,_ = cv2.connectedComponentsWithStats(mask,8)
        ids = list(range(1,int(labels.max())+1))
    regions=[]
    for i in ids:
        region = np.uint8(labels==i)*255
        area = int(np.count_nonzero(region))
        if not (min_area <= area <= max_area):
            continue
        contours,_ = cv2.findContours(region,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
        if not contours: continue
        c=max(contours,key=cv2.contourArea)
        x,y,w,h=cv2.boundingRect(c)
        m=cv2.moments(c)
        cx=int(m['m10']/m['m00']) if m['m00'] else x+w//2
        cy=int(m['m01']/m['m00']) if m['m00'] else y+h//2
        regions.append((c,area,cx,cy,x,y,w,h))
    regions.sort(key=lambda r:(r[3],r[2]))
    return regions

def choose_method(rgb, threshold, opening, closing, min_area, max_area):
    candidates = METHODS[1:]
    total=rgb.shape[0]*rgb.shape[1]
    best=None
    for method in candidates:
        t=threshold if method not in ('밝은 객체','어두운 객체') else 140
        m=clean_mask(build_mask(rgb,method,t,3),opening,closing)
        regions=components(m,min_area,max_area,False,0.4)
        coverage=sum(r[1] for r in regions)/total
        # Reject methods that mostly segment the background or tiny noise.
        if not regions or coverage < .001 or coverage > .65: continue
        sizes=np.array([r[1] for r in regions],dtype=float)
        score=min(len(regions),120) + 12*min(coverage,.3) - 6*np.std(np.log1p(sizes))
        if best is None or score>best[0]: best=(score,method)
    return best[1] if best else '색상 채도 (HSV)'

with st.sidebar:
    st.header('분석 설정')
    method=st.selectbox('객체 분리 방식',METHODS)
    threshold=st.slider('임계값',0,255,55,5,help='밝기 모드는 보통 100~200, 색상 모드는 30~100에서 시작')
    min_area=st.slider('최소 객체 면적 (픽셀)',10,10000,120,10)
    max_area=st.number_input('최대 객체 면적 (픽셀)',min_value=100,value=1000000,step=1000)
    opening=st.slider('작은 잡음 제거',0,6,1)
    closing=st.slider('객체 내부 구멍 메우기',0,8,1)
    split=st.checkbox('붙어 있는 객체 분리 (Watershed)',value=False)
    peak_ratio=st.slider('Watershed 분리 강도',0.1,0.9,0.4,0.05,disabled=not split)
    st.info('자동 추천은 간단한 휴리스틱으로, 모든 사진에 최적의 방법을 보장하지 않습니다. 결과 마스크를 확인해 주세요.')

uploaded=st.file_uploader('분석할 사진을 업로드하세요',type=['jpg','jpeg','png','webp'])
if uploaded is None:
    st.info('이미지를 업로드하면 원본, 분리 마스크, 번호가 표시된 검출 결과를 확인할 수 있습니다.')
    st.stop()

im=ImageOps.exif_transpose(Image.open(uploaded)).convert('RGB')
im.thumbnail((1400,1400))
rgb=np.asarray(im)
selected=choose_method(rgb,threshold,opening,closing,min_area,max_area) if method=='자동 추천' else method
mask=clean_mask(build_mask(rgb,selected,threshold,3),opening,closing)
regions=components(mask,min_area,int(max_area),split,peak_ratio)
annotated=rgb.copy()
rows=[]
for n,(contour,area,cx,cy,x,y,w,h) in enumerate(regions,1):
    cv2.drawContours(annotated,[contour],-1,(255,50,65),2)
    cv2.circle(annotated,(cx,cy),3,(0,255,0),-1)
    cv2.putText(annotated,str(n),(max(0,cx-12),max(18,cy-8)),cv2.FONT_HERSHEY_SIMPLEX,.55,(0,0,0),3,cv2.LINE_AA)
    cv2.putText(annotated,str(n),(max(0,cx-12),max(18,cy-8)),cv2.FONT_HERSHEY_SIMPLEX,.55,(255,255,0),1,cv2.LINE_AA)
    rows.append({'번호':n,'면적(px)':area,'중심 X':cx,'중심 Y':cy,'너비':w,'높이':h})

st.metric('검출된 객체 수',f'{len(regions)}개')
st.write(f'적용된 분석 방식: **{selected}**')
col1,col2,col3=st.columns(3)
with col1: st.image(rgb,caption='원본 이미지',use_container_width=True)
with col2: st.image(mask,caption='객체 분리 마스크 (흰색 = 검출)',use_container_width=True)
with col3: st.image(annotated,caption='검출 결과 및 객체 번호',use_container_width=True)

buf=io.BytesIO()
Image.fromarray(annotated).save(buf,format='PNG')
st.download_button('📥 검출 결과 이미지 다운로드',buf.getvalue(),'object_counting_result.png','image/png')
if rows:
    df=pd.DataFrame(rows)
    with st.expander('객체별 측정 결과'):
        st.dataframe(df,use_container_width=True,hide_index=True)
        st.download_button('CSV 다운로드',df.to_csv(index=False).encode('utf-8-sig'),'object_measurements.csv','text/csv')
else:
    st.warning('검출된 객체가 없습니다. 분석 방식, 임계값, 최소 면적을 조절해 주세요.')
st.caption('주의: 색상/밝기 기반 영상처리 결과이며, 겹친 물체나 복잡한 배경에서는 실제 개수와 다를 수 있습니다.')
