#!/usr/bin/env python3
"""
Simple script to convert PROJECT_EXPLANATION.md to HTML
"""

import os

def create_html():
    """Convert markdown to HTML"""
    
    if not os.path.exists("PROJECT_EXPLANATION.md"):
        print("❌ PROJECT_EXPLANATION.md not found!")
        return False
    
    print("🔄 Converting PROJECT_EXPLANATION.md to HTML...")
    
    # Read markdown file
    with open("PROJECT_EXPLANATION.md", "r", encoding="utf-8") as f:
        md_content = f.read()
    
    # Simple markdown to HTML conversion
    html_content = md_content
    
    # Convert headers
    html_content = html_content.replace('# ', '<h1>').replace('\n# ', '</h1>\n<h1>')
    html_content = html_content.replace('## ', '<h2>').replace('\n## ', '</h2>\n<h2>')
    html_content = html_content.replace('### ', '<h3>').replace('\n### ', '</h3>\n<h3>')
    html_content = html_content.replace('#### ', '<h4>').replace('\n#### ', '</h4>\n<h4>')
    
    # Close headers
    html_content = html_content.replace('<h1>', '<h1>').replace('<h2>', '<h2>').replace('<h3>', '<h3>').replace('<h4>', '<h4>')
    
    # Convert code blocks
    html_content = html_content.replace('```python', '<pre><code class="python">')
    html_content = html_content.replace('```', '</code></pre>')
    
    # Convert inline code
    html_content = html_content.replace('`', '<code>').replace('<code>', '<code>', 1)
    
    # Convert bold
    html_content = html_content.replace('**', '<strong>').replace('<strong>', '<strong>', 1)
    
    # Convert line breaks
    html_content = html_content.replace('\n', '<br>\n')
    
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
        body {{
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            line-height: 1.6;
            margin: 40px;
            color: #333;
            max-width: 1200px;
            margin: 0 auto;
            padding: 20px;
        }}
        h1 {{
            color: #1f77b4;
            border-bottom: 3px solid #1f77b4;
            padding-bottom: 10px;
            font-size: 2.5em;
        }}
        h2 {{
            color: #2ca02c;
            margin-top: 30px;
            font-size: 2em;
        }}
        h3 {{
            color: #d62728;
            margin-top: 25px;
            font-size: 1.5em;
        }}
        h4 {{
            color: #ff7f0e;
            margin-top: 20px;
            font-size: 1.2em;
        }}
        code {{
            background-color: #f4f4f4;
            padding: 2px 4px;
            border-radius: 3px;
            font-family: 'Courier New', monospace;
            color: #d63384;
        }}
        pre {{
            background-color: #f8f9fa;
            padding: 15px;
            border-radius: 5px;
            overflow-x: auto;
            border-left: 4px solid #1f77b4;
        }}
        pre code {{
            background-color: transparent;
            padding: 0;
            color: #333;
        }}
        table {{
            border-collapse: collapse;
            width: 100%;
            margin: 20px 0;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }}
        th, td {{
            border: 1px solid #ddd;
            padding: 12px;
            text-align: left;
        }}
        th {{
            background-color: #f2f2f2;
            font-weight: bold;
            color: #333;
        }}
        tr:nth-child(even) {{
            background-color: #f9f9f9;
        }}
        strong {{
            color: #d63384;
            font-weight: bold;
        }}
        .emoji {{
            font-size: 1.2em;
        }}
        ul, ol {{
            margin: 15px 0;
            padding-left: 30px;
        }}
        li {{
            margin: 8px 0;
        }}
        blockquote {{
            border-left: 4px solid #1f77b4;
            margin: 20px 0;
            padding: 10px 20px;
            background-color: #f8f9fa;
            font-style: italic;
        }}
        @media print {{
            body {{
                margin: 0;
                padding: 20px;
            }}
            h1, h2, h3, h4 {{
                page-break-after: avoid;
            }}
            pre, table {{
                page-break-inside: avoid;
            }}
        }}
    </style>
</head>
<body>
    {html_content}
</body>
</html>
        """)
    
    print(f"✅ HTML file created: {html_file}")
    print("💡 You can:")
    print("   1. Open this HTML file in a browser")
    print("   2. Press Ctrl+P to print")
    print("   3. Select 'Save as PDF'")
    print("   4. Or use online converters like: https://html-to-pdf.net/")
    
    return True

def main():
    """Main function"""
    print("🚚 Delivery Duration Prediction Project - HTML Generator")
    print("=" * 60)
    
    if create_html():
        print("\n🎉 HTML generation completed!")
        print("📄 Check the current directory for PROJECT_EXPLANATION.html")
    else:
        print("\n❌ HTML generation failed")

if __name__ == "__main__":
    main()
