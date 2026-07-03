"""
HMM 기반 국면 분류 알파 패키지

기획서 참조: SNU Quant/HMM_regime_plan.md

구성:
- features/      : 데이터 리샘플링, 지표 계산, 윈도우 피처 추출
- regime/        : HMM 라벨링 + (X, y) 데이터셋 빌더
- classifiers/   : Phase 3 — Base Classifier (ADX, R² 등)
- meta_model/    : Phase 3 — 최종 국면 확률 산출 메타 모델
- position/      : Phase 4 — 확률 → 포지션 비중 변환
- scripts/       : 검증 / 시각화 실행 스크립트
- config.py      : 모든 튜닝 가능 변수 모음 (Phase 1~4 공통)
"""
