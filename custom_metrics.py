from collections import Counter

def compute_f1(prediction: str, ground_truth: str) -> float:
    """
    두 문자열에 대해 단어 단위의 F1 점수를 계산합니다.
    - prediction: 모델의 예측 문자열
    - ground_truth: 정답 문자열

    F1 점수는 다음과 같이 계산합니다.
      precision = (예측한 단어 중 정답과 겹치는 단어의 개수) / (예측한 단어의 총 개수)
      recall = (정답 단어 중 예측과 겹치는 단어의 개수) / (정답 단어의 총 개수)
      F1 = 2 * precision * recall / (precision + recall)
    
    단, 예측과 정답이 모두 빈 문자열이면 1.0, 한쪽만 빈 경우에는 0.0을 반환합니다.
    """
    # 소문자로 변환하고 공백을 기준으로 토큰화합니다.
    pred_tokens = prediction.lower().split()
    gt_tokens = ground_truth.lower().split()

    # 둘 다 빈 문자열인 경우
    if len(pred_tokens) == 0 and len(gt_tokens) == 0:
        return 1.0
    # 한쪽만 빈 경우
    if len(pred_tokens) == 0 or len(gt_tokens) == 0:
        return 0.0

    # 토큰의 등장 횟수를 계산합니다.
    pred_counter = Counter(pred_tokens)
    gt_counter = Counter(gt_tokens)
    
    # 두 카운터에서 공통으로 등장하는 단어의 최소 등장 횟수를 합산합니다.
    common = sum(min(pred_counter[token], gt_counter[token]) for token in pred_counter)
    
    if common == 0:
        return 0.0

    precision = common / len(pred_tokens)
    recall = common / len(gt_tokens)
    f1 = 2 * precision * recall / (precision + recall)
    return f1


def levenshtein_distance(seq1: list, seq2: list) -> int:
    """
    두 시퀀스(seq1, seq2) 사이의 Levenshtein 거리(편집 거리)를 계산합니다.
    편집 거리는 삽입, 삭제, 치환 연산의 최소 횟수입니다.
    """
    n = len(seq1)
    m = len(seq2)
    
    # DP 테이블 초기화 (크기: (n+1) x (m+1))
    dp = [[0] * (m + 1) for _ in range(n + 1)]
    for i in range(n + 1):
        dp[i][0] = i  # seq1의 i개 요소를 모두 삭제하는 비용
    for j in range(m + 1):
        dp[0][j] = j  # seq2의 j개 요소를 모두 삽입하는 비용
    
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            if seq1[i - 1] == seq2[j - 1]:
                cost = 0
            else:
                cost = 1
            dp[i][j] = min(
                dp[i - 1][j] + 1,      # 삭제
                dp[i][j - 1] + 1,      # 삽입
                dp[i - 1][j - 1] + cost  # 치환
            )
    return dp[n][m]


def compute_per(prediction: str, ground_truth: str) -> float:
    """
    음성 인식에서 Phone Error Rate(PER)를 계산합니다.
    - prediction: 모델의 예측 전화 시퀀스 (예: "AH B K")
    - ground_truth: 정답 전화 시퀀스 (예: "AH B K")
    
    PER는 편집 거리(Levenshtein distance)를 정답 전화 수로 나누어 계산합니다.
    """
    # 공백 기준으로 전화 단위를 분리합니다.
    pred_tokens = prediction.split()
    gt_tokens = ground_truth.split()
    
    # 정답이 없는 경우 PER는 0으로 처리합니다.
    if len(gt_tokens) == 0:
        return 0.0
    
    edit_distance = levenshtein_distance(pred_tokens, gt_tokens)
    per = edit_distance / len(gt_tokens)
    return per
