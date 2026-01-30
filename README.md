

---

# 👁 Visual Acuity Prediction from Fundus Images

---
**문제 제기 → 데이터 재설계 → 모델 선택 → 아키텍처 제안 → 학습 전략 → 실험 검증 → 기여 → 결론**

## 🧠 Abstract

본 연구는 안저(Fundus) 이미지를 활용한 시력 예측 딥러닝 모델 개발 과정에서 기존 연구 데이터셋에 존재하던 **데이터 누수(Data Leakage) 및 환자 단위 분리 미흡 문제**를 발견한 것에서 시작되었다.

단순 성능 개선이 아닌, 의료 AI에서 필수적인 **데이터 재설계(Data Re-engineering)**를 수행하고, 클래스 간 특징 모호성(inter-class feature ambiguity)을 완화하기 위해 **Hierarchical Classification 구조**를 제안하였다.

> Hierarchical classification mitigates inter-class feature ambiguity in visual acuity prediction.

---

# 🔎 1. Research Motivation

기존 시력 예측 연구에서 다음과 같은 구조적 문제를 확인하였다:

* 동일 환자 이미지가 train/validation/test에 동시에 포함
* 심각한 클래스 불균형
* 클래스 간 특징 차이에 대한 분석 부족
* 높은 Train Accuracy 대비 낮은 Validation Accuracy (Overfitting 의심)

이는 실제 임상 환경에서 모델이 일반화되지 않을 가능성을 의미한다.

따라서 본 연구의 목표는:

> 데이터 구조를 재설계하여 신뢰 가능한 의료 AI 모델을 구축하는 것

---

# 🗂 2. Data Re-Engineering

## 2.1 Patient-wise Split

* 환자 고유 ID 기반 데이터 분리
* Train / Validation / Test 간 중복 제거
* 동일 환자 데이터가 서로 다른 세트에 포함되지 않도록 설계

✔ Data Leakage 완전 차단
✔ 실제 임상 환경과 유사한 일반화 검증

---

## 2.2 Dataset Balancing Strategy

의료 데이터 특성상 클래스별 환자 수 및 이미지 수 편차가 존재하였다.
각 클래스별 환자 수와 1인당 이미지 개수(최소/최대/평균)를 분석한 결과:

* 일부 환자는 100장 이상의 이미지를 보유
* 일부 환자는 1장의 이미지만 보유
* 클래스별 환자 수 역시 불균형 존재

이로 인해 모델이 특정 환자의 특성을 과도하게 학습할 위험이 있었다.

### ✔ Generalization 개선 전략

* 한 환자당 **최대 10장의 이미지만 사용**
* 최대한 다양한 환자 데이터를 포함하도록 설계
* 과다 샘플 환자에 대한 과적합 방지
* 모델이 “환자 특징”이 아닌 “질병 특징”을 학습하도록 유도

이를 통해:

> 모델의 일반화 성능을 구조적으로 개선

---

## 2.3 Class Mapping

원본 11-class 데이터를 4-class 구조로 재설계:

```
0 → Class 0
1,2 → Class 1
3,4,5,6,7 → Class 2
8,9,10 → Class 3
```

목적:

* 극단값 완화
* 분포 안정화
* 계층적 분류 구조 설계

---

## 2.4 Balanced Sampling

* 단계별 분류마다 별도 샘플링 전략 적용
* 특정 클래스 편향 최소화
* 학습 데이터 분포 안정화

---

# 🔍 3. Model Selection Strategy

전체 4-class 실험 이전에,
**클래스 간 차이가 비교적 뚜렷하면서 데이터 수가 유사한 0과 5 클래스를 대상으로 사전 실험을 수행하였다.**

### 📌 목적

* 모델 feature extraction 능력 비교
* 데이터 수 편향 없는 공정한 평가
* Backbone 선택 기준 마련

---

## 📊 모델 비교 결과

| Metric    | ViT    | VGG19  | VGG16  | Xception   |
| --------- | ------ | ------ | ------ | ---------- |
| Precision | 0.7183 | 0.7543 | 0.7872 | **0.8379** |
| Recall    | 0.7050 | 0.8391 | 0.8169 | **0.9065** |
| F1 Score  | 0.7116 | 0.7946 | 0.8018 | **0.8709** |
| AUC       | 0.80   | 0.86   | 0.87   | **0.92**   |

### ✅ 최종 선택 모델: **Xception**

선택 근거:

* 가장 높은 F1 Score
* 가장 높은 AUC
* 높은 Recall (의료 AI에서 FN 최소화 중요)

---

# 🧠 4. Model Architecture & Training Setup

## Backbone

* Fine-tuned **Xception**
* Input size: **299 × 299**

---

## ⚙ Training Parameters

* Optimizer: **Adam**
* Learning rate: **1e-4**
* Batch size: **64**
* Epoch: **25 (Early Stopping 적용)**
* Train : Validation : Test = **0.8 : 0.1 : 0.1**
* Mixed Precision Training 사용

---

## 🎯 Training Strategy

* Early Stopping → 과적합 방지
* Mixed Precision → 연산 효율 개선
* Patient-wise Split → 일반화 검증
* Per-patient Image Cap (max 10) → 환자 편향 최소화


---

# 🔁 5. Hierarchical Classification Framework

```
Fundus Image
      ↓
Fine-tuned Xception
      ↓
Level 1: (Class 0,1) vs (Class 2,3)
      ↓
Level 2A: Class 0 vs Class 1
Level 2B: Class 2 vs Class 3
      ↓
Final 4-Class Prediction
```

---

# 🎯 6. Why Hierarchical Classification?

Flat 4-class classification은 유사 클래스 간 특징 혼동을 유발할 가능성이 높다.

Hierarchical 구조는:

* 큰 특징 차이부터 우선 분리
* 점진적 decision boundary 형성
* inter-class ambiguity 완화
* 보다 안정적인 discriminative boundary 형성

---

# 📈 7. Experimental Validation

각 단계에서:

* Precision
* Recall
* F1 Score
* Confusion Matrix
* ROC Curve

를 통해 성능 검증 수행.

---

## 📊 Performance

| Stage    | Task           | F1 Score |
| -------- | -------------- | -------- |
| Level 1  | (0,1) vs (2,3) | 0.9146   |
| Level 2A | 0 vs 1         | 0.7122   |
| Level 2B | 2 vs 3         | 0.7765   |

---

# 🧪 8. Training Strategy

* Early Stopping → 과적합 방지
* Mixed Precision → 연산 효율 개선
* Balanced Sampling → 편향 감소
* Patient-wise Split → 일반화 검증

---

# 📊 9. Key Contributions

✔ 데이터 누수 문제 구조적 해결
✔ 환자 단위 일반화 설계
✔ 클래스 간 특징 모호성 분석
✔ Backbone 비교 기반 모델 선택
✔ Hierarchical Classification 구조 제안

---

# ⚠️ 데이터 사용 및 공개 제한 안내

본 연구는 모 대학병원으로부터 연구 목적 하에 제공받은 임상 데이터를 기반으로 수행되었습니다.

의료 데이터 보호 정책에 따라:

* 원본 데이터는 공개되지 않습니다.
* 전처리 데이터는 공개되지 않습니다.
* 학습된 모델 weight 파일은 공개하지 않습니다.

본 저장소는 연구 방법론, 모델 구조 및 성능 결과 공유 목적입니다.

---

# 🔮 Future Work

* Regression 기반 시력 수치 예측 확장
* Multimodal 학습 (영상 + 임상 정보)
* Explainable AI (Grad-CAM)
* 11-class 세부 분류 복원 실험

---

# 🏁 Conclusion

본 연구는 단순 모델 성능 개선이 아닌,
**데이터 재설계 기반 의료 AI 신뢰성 확보**에 초점을 맞추었다.

계층적 분류 구조를 통해 클래스 간 특징 모호성을 완화하고,
보다 일반화 가능한 시력 예측 모델을 구축하였다.

---

이 구조가 지금 가장 깔끔하고 “연구 논리 흐름”에 맞아.

원하면 다음 단계로:

* 🔬 논문 스타일로 더 academic하게 톤 올리기
* 💼 기업용으로 30% 줄여서 간결하게 만들기
* 🌍 완전 영어 논문형 버전 만들기
* 📊 아키텍처 다이어그램 텍스트까지 README용으로 정리

어디까지 끌어올릴지 말해줘 😎
