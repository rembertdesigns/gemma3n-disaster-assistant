from app.inference import run_disaster_analysis
from app.preprocessing import preprocess_input
import sys

def main():
    print("\n📦 Disaster Response & Recovery Assistant (Gemma 3n)")
    print("-----------------------------------------------------")

    if len(sys.argv) < 2:
        raw_input = input("Enter a report (or file path to image/audio): ").strip()
    else:
        raw_input = sys.argv[1]

    print(f"\n🧠 Processing input: {raw_input}")

    # Preprocess input (text, image, or audio)
    processed = preprocess_input(raw_input)

    if processed["type"] == "text":
        print(f"\n📨 Transcribed Text:\n{processed['content']}")

    # Run AI analysis
    result = run_disaster_analysis(processed)

    # Output result
    print("\n✅ Inference Output:")
    for key, value in result.items():
        print(f"  - {key}: {value}")

    print("\n🛟 Done. Stay safe out there!\n")

if __name__ == "__main__":
    main()

