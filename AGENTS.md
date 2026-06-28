# Agent Instructions

## 기록 지침

- 사용자가 이후 메시지에서 명시적으로 기록, 추가, 정리, 저장을 요청하는 내용은 이 파일에 누적 기록한다.
- 사용자가 지침성 내용을 제공하면 원문 의미를 보존하되, 중복되거나 흩어진 내용은 읽기 쉬운 섹션으로 정리한다.
- 단순 질문, 임시 대화, 실행 결과 설명은 별도 요청이 없는 한 기록하지 않는다.
- 기록 시에는 기존 내용을 불필요하게 삭제하거나 되돌리지 않는다.

## Hugging Face JFrog Pre-Warmer

- 목표는 개발자 Laptop에서 Public JFrog를 경유해 Hugging Face 모델 파일을 미리 `GET`하여 Public JFrog remote cache를 pre-warm하는 것이다.
- 네트워크 구조는 `Hugging Face -> Public JFrog -> Private JFrog -> GPU Instance`이다.
- GPU Instance는 Public JFrog에 접근할 수 없고, 개발자 Laptop은 Public JFrog에 접근할 수 있다.
- 개발자 Laptop은 스토리지 여유가 작으므로 기본 동작은 모델 payload를 디스크에 저장하지 않아야 한다.
- 기본 구현은 `nvidia/GLM-5.2-NVFP4`의 runtime-minimal 파일을 대상으로 한다.
- 기본 GET 대상은 Hugging Face repo의 `resolve/{revision}/{filename}` 경로이며, safetensors shard는 `model-00001-of-00047.safetensors`부터 `model-00047-of-00047.safetensors`까지다.
- 기본 모드는 응답 바디를 chunk 단위로 읽고 즉시 폐기하는 `discard` 방식이다.
- 임시 파일 저장 후 삭제 방식은 선택 옵션으로만 제공하고, 시작 전에 충분한 여유 공간을 확인해야 한다.
- `HEAD` 요청이나 작은 range 요청만으로는 complete artifact cache가 보장되지 않는다고 보고, 기본 pre-warm은 full `GET` 응답 바디를 끝까지 소비한다.
