#!/usr/bin/env python3
"""
Script to convert PROJECT_EXPLANATION.md to PDF format
"""

import os
import subprocess
import sys

def install_requirements():
    """Install required packages for PDF conversion"""
    try:
        import markdown
        import pdfkit
        import weasyprint
        print("✅ Required packages already installed")
        return True
    except ImportError:
        print("📦 Installing required packages...")
        try:
            subprocess.check_call([sys.executable, "-m", "pip", "install", 
                                 "markdown", "pdfkit", "weasyprint"])
            print("✅ Packages installed successfully")
            return True
        except subprocess.CalledProcessError:
            print("❌ Failed to install packages")
            return False

def convert_md_to_pdf():
    """Convert markdown to PDF using multiple methods"""
    
    if not os.path.exists("PROJECT_EXPLANATION.md"):
        print("❌ PROJECT_EXPLANATION.md not found!")
        return False
    
    print("🔄 Converting PROJECT_EXPLANATION.md to PDF...")
    
    # Method 1: Try weasyprint (most reliable)
    try:
        import weasyprint
        from markdown import markdown
        
        # Read markdown file
        with open("PROJECT_EXPLANATION.md", "r", encoding="utf-8") as f:
            md_content = f.read()
        
        # Convert to HTML
        import markdown
        html_content = markdown.markdown(md_content, extensions=['tables', 'codehilite'])
        
        # Add CSS styling
        styled_html = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <meta charset="utf-8">
            <style>
                body {{
                    font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
                    line-height: 1.6;
                    margin: 40px;
                    color: #333;
                }}
                h1 {{
                    color: #1f77b4;
                    border-bottom: 3px solid #1f77b4;
                    padding-bottom: 10px;
                }}
                h2 {{
                    color: #2ca02c;
                    margin-top: 30px;
                }}
                h3 {{
                    color: #d62728;
                    margin-top: 25px;
                }}
                code {{
                    background-color: #f4f4f4;
                    padding: 2px 4px;
                    border-radius: 3px;
                    font-family: 'Courier New', monospace;
                }}
                pre {{
                    background-color: #f4f4f4;
                    padding: 15px;
                    border-radius: 5px;
                    overflow-x: auto;
                }}
                table {{
                    border-collapse: collapse;
                    width: 100%;
                    margin: 20px 0;
                }}
                th, td {{
                    border: 1px solid #ddd;
                    padding: 12px;
                    text-align: left;
                }}
                th {{
                    background-color: #f2f2f2;
                    font-weight: bold;
                }}
                .emoji {{
                    font-size: 1.2em;
                }}
                @page {{
                    margin: 2cm;
                    size: A4;
                }}
            </style>
        </head>
        <body>
            {html_content}
        </body>
        </html>
        """
        
        # Convert to PDF
        weasyprint.HTML(string=styled_html).write_pdf("PROJECT_EXPLANATION.pdf")
        print("✅ PDF created successfully using WeasyPrint: PROJECT_EXPLANATION.pdf")
        return True
        
    except Exception as e:
        print(f"❌ WeasyPrint method failed: {e}")
    
    # Method 2: Try pdfkit (requires wkhtmltopdf)
    try:
        import pdfkit
        from markdown import markdown
        
        # Read markdown file
        with open("PROJECT_EXPLANATION.md", "r", encoding="utf-8") as f:
            md_content = f.read()
        
        # Convert to HTML
        import markdown
        html_content = markdown.markdown(md_content, extensions=['tables', 'codehilite'])
        
        # Add CSS styling
        styled_html = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <meta charset="utf-8">
            <style>
                body {{ font-family: Arial, sans-serif; line-height: 1.6; margin: 40px; }}
                h1 {{ color: #1f77b4; border-bottom: 3px solid #1f77b4; }}
                h2 {{ color: #2ca02c; margin-top: 30px; }}
                h3 {{ color: #d62728; margin-top: 25px; }}
                code {{ background-color: #f4f4f4; padding: 2px 4px; }}
                pre {{ background-color: #f4f4f4; padding: 15px; }}
                table {{ border-collapse: collapse; width: 100%; }}
                th, td {{ border: 1px solid #ddd; padding: 12px; }}
                th {{ background-color: #f2f2f2; }}
            </style>
        </head>
        <body>
            {html_content}
        </body>
        </html>
        """
        
        # Convert to PDF
        pdfkit.from_string(styled_html, "PROJECT_EXPLANATION.pdf")
        print("✅ PDF created successfully using pdfkit: PROJECT_EXPLANATION.pdf")
        return True
        
    except Exception as e:
        print(f"❌ pdfkit method failed: {e}")
    
    # Method 3: Simple HTML to PDF conversion
    try:
        from markdown import markdown
        
        # Read markdown file
        with open("PROJECT_EXPLANATION.md", "r", encoding="utf-8") as f:
            md_content = f.read()
        
        # Convert to HTML
        import markdown
        html_content = markdown.markdown(md_content, extensions=['tables', 'codehilite'])
        
        # Create HTML file
        html_file = "PROJECT_EXPLANATION.html"
        with open(html_file, "w", encoding="utf-8") as f:
            f.write(f"""
            <!DOCTYPE html>
            <html>
            <head>
                <meta charset="utf-8">
                <title>Delivery Duration Prediction Project - Complete Technical Explanation</title>
                <style>
                    body {{ font-family: Arial, sans-serif; line-height: 1.6; margin: 40px; }}
                    h1 {{ color: #1f77b4; border-bottom: 3px solid #1f77b4; }}
                    h2 {{ color: #2ca02c; margin-top: 30px; }}
                    h3 {{ color: #d62728; margin-top: 25px; }}
                    code {{ background-color: #f4f4f4; padding: 2px 4px; }}
                    pre {{ background-color: #f4f4f4; padding: 15px; }}
                    table {{ border-collapse: collapse; width: 100%; }}
                    th, td {{ border: 1px solid #ddd; padding: 12px; }}
                    th {{ background-color: #f2f2f2; }}
                </style>
            </head>
            <body>
                {html_content}
            </body>
            </html>
            """)
        
        print(f"✅ HTML file created: {html_file}")
        print("💡 You can open this HTML file in a browser and print to PDF")
        print("💡 Or use online converters like: https://html-to-pdf.net/")
        return True
        
    except Exception as e:
        print(f"❌ HTML creation failed: {e}")
        return False

def main():
    """Main function"""
    print("🚚 Delivery Duration Prediction Project - PDF Generator")
    print("=" * 60)
    
    # Install requirements
    if not install_requirements():
        print("❌ Cannot proceed without required packages")
        return
    
    # Convert to PDF
    if convert_md_to_pdf():
        print("\n🎉 PDF generation completed!")
        print("📄 Check the current directory for PROJECT_EXPLANATION.pdf")
    else:
        print("\n❌ PDF generation failed")
        print("💡 Try installing wkhtmltopdf manually or use online converters")

if __name__ == "__main__":
    main()
