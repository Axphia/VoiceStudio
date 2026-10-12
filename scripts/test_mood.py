"""Automated test suite for Mood and Prosody Learning Engine."""
import json
import sys
from pathlib import Path

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from mood import MoodAnalyzer, MLMoodClassifier


def test_english_emotions():
    analyzer = MoodAnalyzer()

    # Test Happy
    res_happy = analyzer.analyze("I just got promoted! This is the happiest day of my life!!")
    assert res_happy.primary_emotion in ["joy", "happiness", "excited"], f"Expected joy, got {res_happy.primary_emotion}"
    assert res_happy.valence > 0.0, f"Expected positive valence, got {res_happy.valence}"
    assert res_happy.prosody.speed_ratio >= 1.0, "Expected normal or faster speed for excited/joy"

    # Test Sad / Depressed
    res_sad = analyzer.analyze("I feel so lonely and depressed... everything is falling apart.")
    assert res_sad.primary_emotion in ["sadness", "sad", "grief"], f"Expected sadness, got {res_sad.primary_emotion}"
    assert res_sad.valence < 0.0, f"Expected negative valence, got {res_sad.valence}"
    assert res_sad.prosody.speed_ratio <= 1.0, "Expected slower speed for sadness"

    # Test Angry
    res_angry = analyzer.analyze("Shut up! I hate you, stop annoying me right now!!")
    assert res_angry.primary_emotion in ["anger", "angry"], f"Expected anger, got {res_angry.primary_emotion}"
    assert res_angry.prosody.energy_ratio > 1.0, "Expected high energy for anger"

    print("✔ English emotion tests passed.")


def test_japanese_dialogue():
    analyzer = MoodAnalyzer()

    # Dialogue from todo.txt
    sample_dialogue = """
田中「最近、仕事が忙しすぎて、寝る暇もないよ。」
鈴木「そう言われてみれば、最近顔色が悪いと思っていたところだ。無理をしてまで働く必要はないんじゃない?」
田中「そうしたいのはやまやまだけど、締め切りが迫っている以上、休むわけにはいかないんだ。」
鈴木「体を壊してからでは遅いよ。上司に相談してみたらどう?」
田中「鈴木に言われるまでもなく、自分でも分かってはいるんだ。でも、この状況で弱音を吐くわけにもいかなくて……。」
鈴木「一人で抱え込んでいるようじゃ、いずれ限界が来るよ。頼れる人には頼るべきだと思う。」
""".strip()

    results = analyzer.analyze_dialogue(sample_dialogue)
    assert len(results) == 6, f"Expected 6 lines, got {len(results)}"

    # Tanaka line 1: Exhaustion / Tired
    t1 = results[0]
    assert t1.speaker == "田中"
    assert t1.primary_emotion in ["exhaustion", "tired", "sadness"], f"Expected exhaustion, got {t1.primary_emotion}"
    assert "[tired]" in t1.format_for_fish_speech() or "[sad]" in t1.format_for_fish_speech()

    # Suzuki line 1: Caring / concern / tender
    s1 = results[1]
    assert s1.speaker == "鈴木"

    # Tanaka line 5: 弱音を吐くわけにもいかなくて…… (struggling / exhaustion)
    t3 = results[4]
    assert t3.speaker == "田中"
    assert t3.prosody.breathiness > 0.0, "Expected breathiness for ellipses / sigh"

    print("✔ Japanese dialogue parsing & mood tagging tests passed.")


def test_json_and_tts_tagging():
    analyzer = MoodAnalyzer()
    res = analyzer.analyze("I'm exhausted... worked 20 hours straight.")
    json_str = res.to_json()
    data = json.loads(json_str)

    assert "primary_emotion" in data
    assert "intensity" in data
    assert "prosody" in data
    assert "tts_style" in data
    assert data["prosody"]["speed_ratio"] > 0
    assert data["tts_style"]["fish_audio_text"] != ""

    formatted = analyzer.format_tts_script("田中「最近、仕事が忙しすぎて、寝る暇もないよ。」")
    assert "田中「" in formatted
    print("✔ JSON serialization and TTS tagging tests passed.")


def run_all():
    print("Running Mood & Prosody Learning Engine tests...\n")
    test_english_emotions()
    test_japanese_dialogue()
    test_json_and_tts_tagging()
    print("\n All tests passed successfully!")


if __name__ == "__main__":
    run_all()
