import os
import glob
from nlp_analyzer import AdvancedNLPAnalyzer

def main():
    print("🚀 Starting NLP Analysis...")
    print("=" * 50)
    
    # Initialize the analyzer
    analyzer = AdvancedNLPAnalyzer()
    
    # Check what files are in the data folder
    data_folder = "data"
    
    if not os.path.exists(data_folder):
        print(f"❌ Data folder '{data_folder}' not found!")
        print("Please create a 'data' folder and put your text files there.")
        return
    
    # Find all text files in data folder
    text_files = glob.glob(os.path.join(data_folder, "*.txt"))
    
    if not text_files:
        print(f"❌ No .txt files found in '{data_folder}' folder!")
        print("Please add .txt files to the data folder.")
        return
    
    print(f"📁 Found {len(text_files)} text file(s):")
    for file in text_files:
        print(f"  • {os.path.basename(file)}")
    
    # If multiple files, ask user to choose or combine all
    if len(text_files) == 1:
        selected_file = text_files[0]
        print(f"\n📖 Using: {os.path.basename(selected_file)}")
    else:
        print(f"\n🤔 Multiple files found. Choose an option:")
        print("1. Analyze all files combined")
        print("2. Choose a specific file")
        
        choice = input("Enter your choice (1 or 2): ").strip()
        
        if choice == "1":
            # Combine all files
            combined_text = ""
            for file_path in text_files:
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        combined_text += f.read() + "\n\n"
                    print(f"  ✅ Loaded: {os.path.basename(file_path)}")
                except Exception as e:
                    print(f"  ❌ Error loading {file_path}: {e}")
            
            # Save combined text temporarily
            selected_file = "combined_text"
            analyzer.load_text(combined_text)
        else:
            # Let user choose specific file
            print("\nAvailable files:")
            for i, file_path in enumerate(text_files, 1):
                print(f"{i}. {os.path.basename(file_path)}")
            
            try:
                file_choice = int(input("Enter file number: ")) - 1
                selected_file = text_files[file_choice]
                print(f"\n📖 Selected: {os.path.basename(selected_file)}")
            except (ValueError, IndexError):
                print("❌ Invalid choice. Using first file.")
                selected_file = text_files[0]
    
    # Load the text if not already loaded (for single file case)
    if selected_file != "combined_text":
        try:
            analyzer.load_text(selected_file)
            print(f"✅ Successfully loaded: {os.path.basename(selected_file)}")
        except Exception as e:
            print(f"❌ Error loading file: {e}")
            return
    
    print("\n🔄 Processing text...")
    
    # Run preprocessing
    analyzer.preprocess_text()
    print("✅ Text preprocessing completed")
    
    # Perform analyses
    print("📊 Running basic statistics...")
    stats = analyzer.basic_statistics()
    
    print("💭 Running sentiment analysis...")
    sentiment = analyzer.sentiment_analysis()
    
    print("🎯 Running topic modeling...")
    topics = analyzer.topic_modeling(n_topics=5)
    
    print("🎨 Generating visualizations...")
    analyzer.create_visualizations(save_plots=True, output_dir="./results/")
    
    print("📝 Generating comprehensive report...")
    report = analyzer.generate_report("./results/nlp_analysis_report.txt")
    
    print("\n" + "=" * 50)
    print("🎉 Analysis Complete!")
    print("📁 Results saved in './results/' folder:")
    print("  • nlp_analysis_dashboard.png - Main visualization dashboard")
    print("  • interactive_topics.html - Interactive topic analysis")
    print("  • nlp_analysis_report.txt - Detailed text report")
    print("\n📈 Quick Summary:")
    print(f"  • Total words: {stats['total_words']:,}")
    print(f"  • Total sentences: {stats['total_sentences']:,}")
    print(f"  • Overall sentiment: {sentiment['overall_vader']['compound']:.3f}")
    print(f"  • Topics found: {len(topics['topics'])}")

if __name__ == "__main__":
    # Create results directory if it doesn't exist
    os.makedirs("results", exist_ok=True)
    
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n⚠️  Analysis interrupted by user")
    except Exception as e:
        print(f"\n❌ An error occurred: {e}")
        print("Please check your data files and try again.")