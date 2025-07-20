# requirements.txt
"""
openai==1.3.0
streamlit==1.28.0
python-docx==0.8.11
PyPDF2==3.0.1
nltk==3.8.1
scikit-learn==1.3.0
requests==2.31.0
beautifulsoup4==4.12.2
"""

# main.py
import streamlit as st
import openai
import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
import PyPDF2
from docx import Document
import io

# Download required NLTK data
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')
    
try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')

class CVOptimizer:
    def __init__(self, openai_api_key):
        self.client = openai.OpenAI(api_key=openai_api_key)
        self.stop_words = set(stopwords.words('english'))
    
    def extract_text_from_pdf(self, pdf_file):
        """Extract text from uploaded PDF file"""
        try:
            pdf_reader = PyPDF2.PdfReader(pdf_file)
            text = ""
            for page in pdf_reader.pages:
                text += page.extract_text()
            return text
        except Exception as e:
            st.error(f"Error reading PDF: {str(e)}")
            return ""
    
    def extract_text_from_docx(self, docx_file):
        """Extract text from uploaded DOCX file"""
        try:
            doc = Document(docx_file)
            text = ""
            for paragraph in doc.paragraphs:
                text += paragraph.text + "\n"
            return text
        except Exception as e:
            st.error(f"Error reading DOCX: {str(e)}")
            return ""
    
    def extract_keywords(self, job_desc, max_keywords=20):
        """Extract keywords from job description using TF-IDF"""
        try:
            # Clean and tokenize text
            words = word_tokenize(job_desc.lower())
            words = [word for word in words if word.isalpha() and word not in self.stop_words and len(word) > 2]
            
            # Use TF-IDF to find important terms
            text_for_tfidf = [' '.join(words)]
            vectorizer = TfidfVectorizer(max_features=max_keywords, ngram_range=(1, 2))
            
            try:
                tfidf_matrix = vectorizer.fit_transform(text_for_tfidf)
                feature_names = vectorizer.get_feature_names_out()
                return list(feature_names)
            except:
                # Fallback to simple word frequency
                from collections import Counter
                word_freq = Counter(words)
                return [word for word, freq in word_freq.most_common(max_keywords)]
                
        except Exception as e:
            st.error(f"Error extracting keywords: {str(e)}")
            return []
    
    def extract_company_name(self, job_desc):
        """Extract company name from job description using regex"""
        patterns = [
            r"at\s+([A-Z][a-zA-Z\s&\.]+?)(?:\s+is|\s+seeks|\s+looking|\.|,|\n)",
            r"([A-Z][a-zA-Z\s&\.]+?)\s+is\s+(?:seeking|looking|hiring)",
            r"Join\s+([A-Z][a-zA-Z\s&\.]+?)(?:\s+as|\s+and)",
            r"Company:\s*([A-Z][a-zA-Z\s&\.]+?)(?:\n|$)",
            r"Position:\s*.+?\s+at\s+([A-Z][a-zA-Z\s&\.]+?)(?:\n|$)",
        ]
        
        for pattern in patterns:
            match = re.search(pattern, job_desc, re.IGNORECASE | re.MULTILINE)
            if match:
                company = match.group(1).strip()
                # Clean up common issues
                company = re.sub(r'\s+', ' ', company)  # Multiple spaces
                company = company.rstrip('.,!?')  # Trailing punctuation
                if len(company) > 3 and len(company) < 50:  # Reasonable length
                    return company
        
        return None
    
    def generate_prompt(self, cv_text, job_desc, voice_style, industry):
        """Generate the optimized prompt for LLM"""
        keywords = self.extract_keywords(job_desc)
        keyword_text = ", ".join(keywords)
        company_name = self.extract_company_name(job_desc)
        
        voice_instructions = {
            "Formal": "Use professional, traditional language with formal tone. Focus on precision, structure, and conventional business terminology. Avoid contractions and casual phrases.",
            "Confident": "Use assertive, results-driven language that emphasizes leadership, achievements, and measurable impact. Lead with action verbs and quantify accomplishments.",
            "Friendly": "Use warm, approachable language while maintaining professionalism. Write conversationally but competently, showing personality alongside qualifications."
        }
        
        industry_focus = {
            "Tech": "Prioritize technical proficiencies, development methodologies, system architectures, coding languages, and quantifiable project outcomes. Include relevant certifications and tools.",
            "Marketing": "Emphasize campaign performance metrics, audience growth, ROI, brand impact, cross-channel strategies, and creative problem-solving. Include specific platforms and tools used.",
            "UX": "Highlight user research methodologies, design thinking processes, usability testing, prototyping tools, accessibility considerations, and user satisfaction improvements.",
            "Data": "Focus on analytical methodologies, statistical modeling, data visualization, programming languages (SQL, Python, R), database management, and business intelligence impact.",
            "Finance": "Emphasize financial modeling, risk analysis, regulatory compliance, audit experience, and cost optimization. Include relevant certifications (CFA, CPA, etc.).",
            "Healthcare": "Highlight patient outcomes, compliance with healthcare regulations, clinical experience, and healthcare technology proficiency.",
            "Consulting": "Focus on client management, problem-solving methodologies, project delivery, and measurable business impact across industries."
        }
        
        return f"""
You are a senior career strategist and ATS optimization expert with 15+ years of experience helping candidates land roles at Fortune 500 companies and top startups.

## CONTEXT
A job seeker needs their application materials optimized for a specific role. You must create materials that achieve a 90%+ ATS compatibility score while compelling human reviewers to schedule interviews.

## COMPANY RESEARCH REQUIREMENT
Company: {company_name if company_name else '[Extract from job description]'}
Before writing the cover letter, research and incorporate:
- Company mission, values, and recent developments
- Industry challenges they're facing
- Their competitive advantages
- Recent news, product launches, or achievements

## YOUR TASKS

### 1. CV OPTIMIZATION & ATS SCORING
Rewrite the CV following these requirements:
- **ATS Compliance**: Use standard section headers, consistent formatting, and keyword density of 2-3%
- **Relevance Matching**: Prioritize experiences that directly match job requirements (aim for 80%+ overlap)
- **Impact Quantification**: Include specific metrics, percentages, ROI, and measurable outcomes for every role
- **Keyword Integration**: Naturally incorporate 15-20 relevant terms from the job description
- **Format Optimization**: Use bullet points, clear hierarchy, standard fonts, and ATS-friendly structure

### 2. RESEARCH-DRIVEN COVER LETTER
Write a compelling cover letter that:
- **Company-Specific Opening**: Reference specific company initiatives, values, or recent news
- **Problem-Solution Fit**: Identify a key challenge the company faces and position yourself as the solution
- **Quantified Value Proposition**: Lead with your most impressive, relevant achievement
- **Professional Close**: Clear call-to-action with next steps
- **Length**: 250-350 words (3-4 tight paragraphs)

## STYLE GUIDELINES
**Voice**: {voice_style}
{voice_instructions.get(voice_style, "Use professional tone appropriate for the role level.")}

**Industry Focus**: {industry}
{industry_focus.get(industry, "Focus on transferable skills and relevant experience for this industry.")}

## SOURCE MATERIALS

### ORIGINAL CV:
{cv_text}

### TARGET JOB DESCRIPTION:
{job_desc}

### EXTRACTED KEYWORDS:
{keyword_text}

---

## DELIVERABLES
Provide your response in exactly these sections:

**[ATS_COMPATIBILITY_SCORE]**
Score: X/100
Key Improvements Made:
- [List 3-5 specific optimizations]

**[OPTIMIZED_CV]**
[Your rewritten CV here - use clear formatting with consistent bullet points]

**[COVER_LETTER]**
[Your tailored cover letter with specific company research integration]

**[KEYWORD_OPTIMIZATION_REPORT]**
Keywords Successfully Integrated: [List with frequency]
Keyword Density: X%
ATS Keyword Match Rate: X%

**[SKILLS_GAP_ANALYSIS]**
Missing Skills: [List with learning resources]
Transferable Skills: [Skills to emphasize more]
Certification Recommendations: [If applicable]

## QUALITY ASSURANCE CHECKLIST
Before finalizing, verify:
✓ All content truthful and based on original CV
✓ Company research is accurate and recent
✓ Keywords flow naturally (no stuffing)
✓ Quantifiable results included where possible
✓ Industry terminology used appropriately
✓ ATS compatibility score justified
"""

    def optimize_cv(self, cv_text, job_desc, voice_style, industry):
        """Send prompt to OpenAI and get optimized CV"""
        try:
            prompt = self.generate_prompt(cv_text, job_desc, voice_style, industry)
            
            response = self.client.chat.completions.create(
                model="gpt-4",  # or "gpt-3.5-turbo" for lower cost
                messages=[
                    {"role": "system", "content": "You are an expert career strategist and ATS optimization specialist."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=3000,
                temperature=0.7
            )
            
            return response.choices[0].message.content
            
        except Exception as e:
            st.error(f"Error calling OpenAI API: {str(e)}")
            return None

def main():
    st.set_page_config(
        page_title="AI CV Optimizer",
        page_icon="📄",
        layout="wide"
    )
    
    st.title("🚀 AI-Powered CV & Cover Letter Optimizer")
    st.markdown("Transform your CV into an ATS-friendly, interview-winning document!")
    
    # Sidebar for configuration
    with st.sidebar:
        st.header("⚙️ Configuration")
        
        # OpenAI API Key
        api_key = st.text_input("OpenAI API Key", type="password", help="Get your API key from https://platform.openai.com/api-keys")
        
        if not api_key:
            st.warning("Please enter your OpenAI API key to continue.")
            st.stop()
        
        # Voice style selection
        voice_style = st.selectbox(
            "Voice Style",
            ["Confident", "Formal", "Friendly"],
            help="Choose the tone for your application materials"
        )
        
        # Industry selection
        industry = st.selectbox(
            "Industry",
            ["Data", "Tech", "Marketing", "UX", "Finance", "Healthcare", "Consulting"],
            help="Select your target industry for optimized content"
        )
    
    # Initialize optimizer
    optimizer = CVOptimizer(api_key)
    
    # Main content area
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.header("📄 Upload Your CV")
        
        # CV upload
        cv_file = st.file_uploader(
            "Choose your CV file",
            type=['pdf', 'docx', 'txt'],
            help="Upload your current CV in PDF, DOCX, or TXT format"
        )
        
        cv_text = ""
        if cv_file is not None:
            if cv_file.type == "application/pdf":
                cv_text = optimizer.extract_text_from_pdf(cv_file)
            elif cv_file.type == "application/vnd.openxmlformats-officedocument.wordprocessingml.document":
                cv_text = optimizer.extract_text_from_docx(cv_file)
            else:  # txt file
                cv_text = str(cv_file.read(), "utf-8")
            
            st.success(f"✅ CV uploaded successfully! ({len(cv_text)} characters)")
            
            # Show preview
            with st.expander("📖 CV Preview"):
                st.text_area("Current CV Content", cv_text, height=200, disabled=True)
    
    with col2:
        st.header("💼 Job Description")
        
        job_desc = st.text_area(
            "Paste the job description here",
            height=300,
            help="Copy and paste the full job description from the job posting"
        )
        
        if job_desc:
            # Extract and show company name
            company_name = optimizer.extract_company_name(job_desc)
            if company_name:
                st.info(f"🏢 Detected Company: **{company_name}**")
            
            # Extract and show keywords
            keywords = optimizer.extract_keywords(job_desc)
            if keywords:
                with st.expander("🎯 Extracted Keywords"):
                    st.write(", ".join(keywords[:10]) + "..." if len(keywords) > 10 else ", ".join(keywords))
    
    # Optimization button
    if st.button("🚀 Optimize CV & Generate Cover Letter", type="primary"):
        if not cv_text:
            st.error("Please upload your CV first!")
        elif not job_desc:
            st.error("Please enter the job description!")
        else:
            with st.spinner("🤖 AI is optimizing your application materials..."):
                result = optimizer.optimize_cv(cv_text, job_desc, voice_style, industry)
                
                if result:
                    st.success("✅ Optimization completed!")
                    
                    # Display results in tabs
                    tabs = st.tabs(["📊 ATS Score", "📄 Optimized CV", "💌 Cover Letter", "📈 Analytics", "🎯 Skills Gap"])
                    
                    # Parse the result sections
                    sections = {}
                    current_section = None
                    current_content = []
                    
                    for line in result.split('\n'):
                        if line.startswith('**[') and line.endswith(']**'):
                            if current_section:
                                sections[current_section] = '\n'.join(current_content)
                            current_section = line.strip('**[]')
                            current_content = []
                        else:
                            current_content.append(line)
                    
                    if current_section:
                        sections[current_section] = '\n'.join(current_content)
                    
                    # Display in tabs
                    with tabs[0]:
                        if 'ATS_COMPATIBILITY_SCORE' in sections:
                            st.markdown("### 🎯 ATS Compatibility Analysis")
                            st.markdown(sections['ATS_COMPATIBILITY_SCORE'])
                    
                    with tabs[1]:
                        if 'OPTIMIZED_CV' in sections:
                            st.markdown("### 📄 Your Optimized CV")
                            st.markdown(sections['OPTIMIZED_CV'])
                            
                            # Download button
                            st.download_button(
                                label="📥 Download Optimized CV",
                                data=sections['OPTIMIZED_CV'],
                                file_name="optimized_cv.txt",
                                mime="text/plain"
                            )
                    
                    with tabs[2]:
                        if 'COVER_LETTER' in sections:
                            st.markdown("### 💌 Your Tailored Cover Letter")
                            st.markdown(sections['COVER_LETTER'])
                            
                            # Download button
                            st.download_button(
                                label="📥 Download Cover Letter",
                                data=sections['COVER_LETTER'],
                                file_name="cover_letter.txt",
                                mime="text/plain"
                            )
                    
                    with tabs[3]:
                        if 'KEYWORD_OPTIMIZATION_REPORT' in sections:
                            st.markdown("### 📈 Keyword Optimization Report")
                            st.markdown(sections['KEYWORD_OPTIMIZATION_REPORT'])
                    
                    with tabs[4]:
                        if 'SKILLS_GAP_ANALYSIS' in sections:
                            st.markdown("### 🎯 Skills Gap Analysis")
                            st.markdown(sections['SKILLS_GAP_ANALYSIS'])
                
                else:
                    st.error("Failed to optimize CV. Please check your API key and try again.")

    # Footer
    st.markdown("---")
    st.markdown("💡 **Pro Tip**: Always review and customize the generated content before applying!")

if __name__ == "__main__":
    main()