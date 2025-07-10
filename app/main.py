# app/main.py - Enhanced Disaster Response & Recovery Assistant with Gemma 3n

import sys
import os
import argparse
import json
from datetime import datetime
from pathlib import Path

# Core imports
from app.inference import run_disaster_analysis, Gemma3nEmergencyProcessor
from app.preprocessing import preprocess_input
from app.audio_transcription import VoiceEmergencyProcessor
from app.adaptive_ai_settings import adaptive_optimizer, optimize_for_emergency
from app.database import get_db, engine
from app.models import Base, CrowdReport, VoiceAnalysis, MultimodalAssessment

def print_banner():
    """Print enhanced banner with Gemma 3n features"""
    print("\n" + "="*80)
    print("🚨 DISASTER RESPONSE & RECOVERY ASSISTANT with Gemma 3n AI")
    print("="*80)
    print("🧠 AI-Powered Emergency Analysis with Multimodal Intelligence")
    print("🎤 Voice Emergency Processing | 📷 Image Analysis | 📝 Text Analysis")
    print("⚡ Adaptive AI Optimization | 🗺️ Real-time Mapping | 📊 Predictive Analytics")
    print("="*80)

def setup_database():
    """Initialize database with all tables"""
    try:
        print("🔧 Initializing database...")
        Base.metadata.create_all(bind=engine)
        print("✅ Database initialized successfully")
        return True
    except Exception as e:
        print(f"❌ Database initialization failed: {e}")
        return False

def optimize_system():
    """Optimize system for emergency response"""
    try:
        print("⚡ Optimizing AI system for emergency response...")
        
        # Optimize for emergency use case
        config = optimize_for_emergency()
        
        print(f"✅ AI optimized: {config.model_variant}")
        print(f"   • Context Window: {config.context_window:,} tokens")
        print(f"   • Precision: {config.precision}")
        print(f"   • Optimization: {config.optimization_level}")
        
        # Get device info
        device_info = adaptive_optimizer.device_caps
        print(f"   • Device: {device_info.cpu_cores} cores, {device_info.memory_gb:.1f}GB RAM")
        print(f"   • GPU: {'Available' if device_info.gpu_available else 'Not available'}")
        
        return config
    except Exception as e:
        print(f"⚠️  System optimization failed: {e}")
        return None

def analyze_text(text_input: str, save_to_db: bool = False) -> dict:
    """Analyze text input with enhanced Gemma 3n processing"""
    
    print(f"\n📝 Analyzing text input...")
    print(f"Input: {text_input[:100]}{'...' if len(text_input) > 100 else ''}")
    
    try:
        # Use enhanced Gemma 3n processor
        processor = Gemma3nEmergencyProcessor()
        
        result = processor.analyze_multimodal_emergency(
            text=text_input,
            context={
                "analysis_mode": "cli",
                "timestamp": datetime.utcnow().isoformat(),
                "source": "command_line"
            }
        )
        
        # Save to database if requested
        if save_to_db:
            try:
                db = next(get_db())
                
                # Create crowd report
                crowd_report = CrowdReport(
                    message=text_input,
                    escalation=result.get("emergency_type", {}).get("primary", "unknown"),
                    severity=int(result.get("severity", {}).get("overall_score", 5)),
                    confidence_score=result.get("severity", {}).get("confidence", 0.0),
                    source="cli_analysis",
                    ai_analysis=result,
                    timestamp=datetime.utcnow().isoformat(),
                    user="CLI_User"
                )
                
                db.add(crowd_report)
                db.commit()
                print(f"✅ Saved to database with ID: {crowd_report.id}")
                
            except Exception as e:
                print(f"⚠️  Failed to save to database: {e}")
        
        return result
        
    except Exception as e:
        print(f"❌ Text analysis failed: {e}")
        return {"error": str(e), "analysis_failed": True}

def analyze_voice(audio_path: str, save_to_db: bool = False) -> dict:
    """Analyze voice input with enhanced processing"""
    
    if not os.path.exists(audio_path):
        return {"error": f"Audio file not found: {audio_path}"}
    
    print(f"\n🎤 Analyzing voice input: {audio_path}")
    
    try:
        # Use enhanced voice processor
        voice_processor = VoiceEmergencyProcessor()
        
        result = voice_processor.process_emergency_call(
            audio_path=audio_path,
            context={
                "analysis_mode": "cli",
                "source": "command_line"
            }
        )
        
        # Save to database if requested
        if save_to_db:
            try:
                db = next(get_db())
                
                # Create voice analysis record
                voice_analysis = VoiceAnalysis(
                    audio_file_path=audio_path,
                    transcript=result.get("transcript", ""),
                    confidence=result.get("confidence", 0.0),
                    urgency_level=result.get("overall_urgency", "unknown"),
                    emergency_type=result.get("gemma_analysis", {}).get("emergency_type", "unknown"),
                    emotional_state=result.get("emotional_state", {}),
                    hazards_detected=result.get("hazards_detected", []),
                    location_extracted=str(result.get("location_info", {})),
                    processing_metadata={
                        "processing_time": result.get("processing_time"),
                        "audio_duration": result.get("audio_duration", 0),
                        "model_used": "gemma_3n_voice_processor"
                    },
                    analyst_id="CLI_System"
                )
                
                db.add(voice_analysis)
                db.commit()
                print(f"✅ Voice analysis saved with ID: {voice_analysis.id}")
                
                # Auto-create crowd report for high urgency
                if result.get("overall_urgency") in ["critical", "high"]:
                    crowd_report = CrowdReport(
                        message=f"VOICE EMERGENCY: {result.get('transcript', '')[:200]}...",
                        escalation=result.get("overall_urgency"),
                        severity=8 if result.get("overall_urgency") == "critical" else 6,
                        source="voice_analysis_system",
                        voice_analysis_id=voice_analysis.id,
                        urgency_detected=result.get("overall_urgency"),
                        timestamp=datetime.utcnow().isoformat(),
                        user="Voice_System"
                    )
                    
                    db.add(crowd_report)
                    db.commit()
                    print(f"✅ Auto-created emergency report: ID {crowd_report.id}")
                
            except Exception as e:
                print(f"⚠️  Failed to save to database: {e}")
        
        return result
        
    except Exception as e:
        print(f"❌ Voice analysis failed: {e}")
        return {"error": str(e), "analysis_failed": True}

def analyze_image(image_path: str, save_to_db: bool = False) -> dict:
    """Analyze image input"""
    
    if not os.path.exists(image_path):
        return {"error": f"Image file not found: {image_path}"}
    
    print(f"\n📷 Analyzing image: {image_path}")
    
    try:
        # Read image data
        with open(image_path, "rb") as f:
            image_data = f.read()
        
        # Use enhanced processor
        processor = Gemma3nEmergencyProcessor()
        
        result = processor.analyze_multimodal_emergency(
            image_data=image_data,
            context={
                "analysis_mode": "cli",
                "image_path": image_path,
                "source": "command_line"
            }
        )
        
        # Save to database if requested
        if save_to_db:
            try:
                db = next(get_db())
                
                # Create multimodal assessment
                assessment = MultimodalAssessment(
                    assessment_type="image_analysis",
                    image_path=image_path,
                    severity_score=result.get("severity", {}).get("overall_score", 0),
                    emergency_type=result.get("emergency_type", {}).get("primary", "unknown"),
                    risk_factors=result.get("immediate_risks", []),
                    resource_requirements=result.get("resource_requirements", {}),
                    ai_confidence=result.get("severity", {}).get("confidence", 0.0),
                    analyst_id="CLI_System"
                )
                
                db.add(assessment)
                db.commit()
                print(f"✅ Image analysis saved with ID: {assessment.id}")
                
            except Exception as e:
                print(f"⚠️  Failed to save to database: {e}")
        
        return result
        
    except Exception as e:
        print(f"❌ Image analysis failed: {e}")
        return {"error": str(e), "analysis_failed": True}

def analyze_multimodal(text: str = None, image_path: str = None, audio_path: str = None, save_to_db: bool = False) -> dict:
    """Comprehensive multimodal analysis"""
    
    print(f"\n🔀 Performing multimodal analysis...")
    print(f"   • Text: {'✓' if text else '✗'}")
    print(f"   • Image: {'✓' if image_path else '✗'}")
    print(f"   • Audio: {'✓' if audio_path else '✗'}")
    
    try:
        # Prepare data
        image_data = None
        audio_data = None
        
        if image_path and os.path.exists(image_path):
            with open(image_path, "rb") as f:
                image_data = f.read()
        
        if audio_path and os.path.exists(audio_path):
            with open(audio_path, "rb") as f:
                audio_data = f.read()
        
        # Use enhanced processor
        processor = Gemma3nEmergencyProcessor("high_performance")
        
        result = processor.analyze_multimodal_emergency(
            text=text,
            image_data=image_data,
            audio_data=audio_data,
            context={
                "analysis_mode": "multimodal_cli",
                "source": "command_line",
                "comprehensive": True
            }
        )
        
        # Save to database if requested
        if save_to_db:
            try:
                db = next(get_db())
                
                # Create comprehensive multimodal assessment
                assessment = MultimodalAssessment(
                    assessment_type="comprehensive_multimodal",
                    text_input=text,
                    image_path=image_path,
                    audio_path=audio_path,
                    severity_score=result.get("severity", {}).get("overall_score", 0),
                    emergency_type=result.get("emergency_type", {}).get("primary", "unknown"),
                    secondary_types=result.get("emergency_type", {}).get("secondary", []),
                    risk_factors=result.get("immediate_risks", []),
                    resource_requirements=result.get("resource_requirements", {}),
                    ai_confidence=result.get("severity", {}).get("confidence", 0.0),
                    processing_metadata=result.get("device_performance", {}),
                    analyst_id="CLI_System"
                )
                
                db.add(assessment)
                db.commit()
                print(f"✅ Multimodal analysis saved with ID: {assessment.id}")
                
            except Exception as e:
                print(f"⚠️  Failed to save to database: {e}")
        
        return result
        
    except Exception as e:
        print(f"❌ Multimodal analysis failed: {e}")
        return {"error": str(e), "analysis_failed": True}

def print_analysis_results(result: dict, analysis_type: str = "Analysis"):
    """Print formatted analysis results"""
    
    if result.get("error"):
        print(f"\n❌ {analysis_type} Error:")
        print(f"   {result['error']}")
        return
    
    print(f"\n✅ {analysis_type} Results:")
    print("="*60)
    
    # Severity information
    severity = result.get("severity", {})
    if severity:
        score = severity.get("overall_score", 0)
        confidence = severity.get("confidence", 0)
        print(f"🚨 Severity: {score}/10 (Confidence: {confidence:.1%})")
        print(f"   Reasoning: {severity.get('reasoning', 'N/A')}")
    
    # Emergency type
    emergency_type = result.get("emergency_type", {})
    if emergency_type:
        primary = emergency_type.get("primary", "Unknown")
        confidence = emergency_type.get("confidence", 0)
        print(f"🏷️  Emergency Type: {primary} (Confidence: {confidence:.1%})")
        
        secondary = emergency_type.get("secondary", [])
        if secondary:
            print(f"   Related: {', '.join(secondary)}")
    
    # Immediate risks
    risks = result.get("immediate_risks", [])
    if risks:
        print(f"\n⚠️  Immediate Risks ({len(risks)}):")
        for i, risk in enumerate(risks[:3], 1):  # Show top 3
            risk_desc = risk.get("risk", "Unknown risk")
            probability = risk.get("probability", 0)
            impact = risk.get("impact", 0)
            print(f"   {i}. {risk_desc}")
            print(f"      Probability: {probability:.1%}, Impact: {impact}/10")
    
    # Priority actions
    actions = result.get("priority_actions", [])
    if actions:
        print(f"\n🎯 Priority Actions ({len(actions)}):")
        for i, action in enumerate(actions[:3], 1):  # Show top 3
            action_desc = action.get("action", "Unknown action")
            priority = action.get("priority", 0)
            timeline = action.get("timeline", "unknown")
            print(f"   {i}. {action_desc}")
            print(f"      Priority: {priority}, Timeline: {timeline}")
    
    # Resource requirements
    resources = result.get("resource_requirements", {})
    if resources:
        print(f"\n🚑 Resource Requirements:")
        personnel = resources.get("personnel", {})
        if personnel:
            for role, count in personnel.items():
                print(f"   • {role.replace('_', ' ').title()}: {count}")
        
        equipment = resources.get("equipment", [])
        if equipment:
            print(f"   • Equipment: {', '.join(equipment)}")
        
        response_time = resources.get("estimated_response_time", "unknown")
        print(f"   • Est. Response Time: {response_time}")
    
    # Voice-specific results
    if "transcript" in result:
        print(f"\n🎤 Voice Analysis:")
        print(f"   Transcript: {result['transcript'][:200]}{'...' if len(result['transcript']) > 200 else ''}")
        print(f"   Urgency: {result.get('overall_urgency', 'unknown')}")
        print(f"   Confidence: {result.get('confidence', 0):.1%}")
        
        emotional_state = result.get("emotional_state", {})
        if emotional_state:
            primary_emotion = emotional_state.get("primary_emotion", "unknown")
            stress_level = emotional_state.get("stress_level", 0)
            print(f"   Emotional State: {primary_emotion} (Stress: {stress_level:.1%})")
    
    # Performance metrics
    device_perf = result.get("device_performance", {})
    if device_perf and any(device_perf.values()):
        print(f"\n⚡ Performance Metrics:")
        cpu = device_perf.get("cpu_usage", 0)
        memory = device_perf.get("memory_usage", 0)
        print(f"   CPU: {cpu:.1f}%, Memory: {memory:.1f}%")
        
        model_device = device_perf.get("model_device", "unknown")
        print(f"   Model Device: {model_device}")

def interactive_mode():
    """Interactive mode for continuous analysis"""
    
    print("\n🔄 Entering Interactive Mode")
    print("Commands: text, voice, image, multimodal, quit, help")
    print("-" * 50)
    
    while True:
        try:
            command = input("\n🤖 Enter command (or 'help'): ").strip().lower()
            
            if command in ["quit", "exit", "q"]:
                print("👋 Goodbye!")
                break
            
            elif command == "help":
                print("""
Available Commands:
  text     - Analyze text input
  voice    - Analyze audio file
  image    - Analyze image file
  multimodal - Comprehensive analysis with multiple inputs
  status   - Show system status
  optimize - Re-optimize system
  quit     - Exit interactive mode
                """)
            
            elif command == "text":
                text_input = input("Enter emergency text: ").strip()
                if text_input:
                    save = input("Save to database? (y/n): ").lower().startswith('y')
                    result = analyze_text(text_input, save_to_db=save)
                    print_analysis_results(result, "Text Analysis")
            
            elif command == "voice":
                audio_path = input("Enter audio file path: ").strip()
                if audio_path:
                    save = input("Save to database? (y/n): ").lower().startswith('y')
                    result = analyze_voice(audio_path, save_to_db=save)
                    print_analysis_results(result, "Voice Analysis")
            
            elif command == "image":
                image_path = input("Enter image file path: ").strip()
                if image_path:
                    save = input("Save to database? (y/n): ").lower().startswith('y')
                    result = analyze_image(image_path, save_to_db=save)
                    print_analysis_results(result, "Image Analysis")
            
            elif command == "multimodal":
                print("Enter inputs (leave blank to skip):")
                text = input("Text: ").strip() or None
                image_path = input("Image path: ").strip() or None
                audio_path = input("Audio path: ").strip() or None
                
                if any([text, image_path, audio_path]):
                    save = input("Save to database? (y/n): ").lower().startswith('y')
                    result = analyze_multimodal(text, image_path, audio_path, save_to_db=save)
                    print_analysis_results(result, "Multimodal Analysis")
                else:
                    print("❌ No inputs provided")
            
            elif command == "status":
                print("\n📊 System Status:")
                perf = adaptive_optimizer.monitor_performance()
                print(f"   CPU: {perf.cpu_usage:.1f}%")
                print(f"   Memory: {perf.memory_usage:.1f}%")
                print(f"   Battery: {perf.battery_level:.1f}%")
                print(f"   Inference Speed: {perf.inference_speed:.1f} tokens/sec")
                
                if adaptive_optimizer.current_config:
                    config = adaptive_optimizer.current_config
                    print(f"   Model: {config.model_variant}")
                    print(f"   Context: {config.context_window:,} tokens")
            
            elif command == "optimize":
                config = optimize_system()
                if config:
                    print("✅ System re-optimized successfully")
            
            else:
                print(f"❌ Unknown command: {command}")
                print("Type 'help' for available commands")
        
        except KeyboardInterrupt:
            print("\n👋 Goodbye!")
            break
        except Exception as e:
            print(f"❌ Error: {e}")

def main():
    """Enhanced main function with comprehensive CLI support"""
    
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Disaster Response Assistant with Gemma 3n AI")
    parser.add_argument("input", nargs="?", help="Input text, file path, or 'interactive'")
    parser.add_argument("--type", choices=["text", "voice", "image", "auto"], default="auto",
                       help="Input type (auto-detect by default)")
    parser.add_argument("--save", action="store_true", help="Save results to database")
    parser.add_argument("--interactive", action="store_true", help="Start interactive mode")
    parser.add_argument("--optimize", action="store_true", help="Show optimization info")
    parser.add_argument("--no-db", action="store_true", help="Skip database initialization")
    parser.add_argument("--multimodal", nargs="+", help="Multiple inputs for multimodal analysis")
    parser.add_argument("--json", action="store_true", help="Output results in JSON format")
    
    args = parser.parse_args()
    
    # Print banner
    print_banner()
    
    # Initialize database unless skipped
    if not args.no_db:
        if not setup_database():
            print("⚠️  Continuing without database...")
    
    # Optimize system
    if args.optimize or not args.input:
        config = optimize_system()
        if args.optimize and not args.input:
            return
    
    # Interactive mode
    if args.interactive or (args.input and args.input.lower() == "interactive"):
        interactive_mode()
        return
    
    # Multimodal analysis
    if args.multimodal:
        if len(args.multimodal) >= 1:
            text = args.multimodal[0] if not os.path.exists(args.multimodal[0]) else None
            image_path = next((f for f in args.multimodal if f.lower().endswith(('.jpg', '.png', '.jpeg'))), None)
            audio_path = next((f for f in args.multimodal if f.lower().endswith(('.wav', '.mp3', '.m4a'))), None)
            
            result = analyze_multimodal(text, image_path, audio_path, save_to_db=args.save)
            
            if args.json:
                print(json.dumps(result, indent=2, default=str))
            else:
                print_analysis_results(result, "Multimodal Analysis")
        return
    
    # Single input analysis
    if not args.input:
        print("\n❓ No input provided. Starting interactive mode...")
        interactive_mode()
        return
    
    # Process single input
    input_str = args.input.strip()
    
    # Determine input type
    if args.type == "auto":
        if os.path.isfile(input_str):
            ext = Path(input_str).suffix.lower()
            if ext in ['.wav', '.mp3', '.m4a', '.ogg']:
                input_type = "voice"
            elif ext in ['.jpg', '.jpeg', '.png', '.bmp', '.gif']:
                input_type = "image"
            else:
                input_type = "text"
        else:
            input_type = "text"
    else:
        input_type = args.type
    
    # Analyze based on type
    if input_type == "text":
        result = analyze_text(input_str, save_to_db=args.save)
        analysis_name = "Text Analysis"
    elif input_type == "voice":
        result = analyze_voice(input_str, save_to_db=args.save)
        analysis_name = "Voice Analysis"
    elif input_type == "image":
        result = analyze_image(input_str, save_to_db=args.save)
        analysis_name = "Image Analysis"
    else:
        result = {"error": f"Unknown input type: {input_type}"}
        analysis_name = "Analysis"
    
    # Output results
    if args.json:
        print(json.dumps(result, indent=2, default=str))
    else:
        print_analysis_results(result, analysis_name)
    
    print(f"\n🛟 Analysis complete. Stay safe out there!")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n👋 Emergency analysis interrupted. Stay safe!")
    except Exception as e:
        print(f"\n❌ Fatal error: {e}")
        sys.exit(1)