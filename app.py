import streamlit as st
import PyPDF2
import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

st.set_page_config(
    page_title="AI Resume Analyzer",
    page_icon="📄",
    layout="wide"
)

st.title("📄 AI Resume Analyzer")
st.caption(
    "AI-powered resume analysis, skill detection and job matching"
)

st.divider()

st.title("📄 AI Resume Analyzer")

st.write(
    "Upload your resume and get an AI-powered analysis "
    "of your skills, resume score and job matches."
)

uploaded_file = st.file_uploader(
    "Upload your Resume (PDF)",
    type=["pdf"]
)

if uploaded_file is not None:

    st.success("Resume uploaded successfully! ✅")
    st.write("File name:", uploaded_file.name)

    # Extract text from PDF
    reader = PyPDF2.PdfReader(uploaded_file)

    resume_text = ""

    for page in reader.pages:
        text = page.extract_text()

        if text:
            resume_text += text + "\n"

    st.subheader("📄 Extracted Resume Text")

    st.text_area(
        "Resume Content",
        resume_text,
        height=300
    )
        # Skills list
    skills = [
        "python",
        "sql",
        "machine learning",
        "pandas",
        "numpy",
        "data analysis",
        "scikit-learn",
        "tensorflow",
        "power bi",
        "excel",
        "flask",
        "git",
        "communication",
        "problem solving"
    ]

    # Detect skills
    resume_lower = resume_text.lower()

    found_skills = []

    for skill in skills:
        if skill in resume_lower:
            found_skills.append(skill)

    st.subheader("🛠️ Skills Detected")

    if found_skills:
        st.write(", ".join(found_skills))
    else:
        st.warning("No predefined skills detected.")
            # -----------------------------
    # RESUME SCORE
    # -----------------------------

    # 1. Skills Score (40)
    skill_score = min(40, len(found_skills) * 4)

    # 2. Structure Score (30)
    sections = {
        "Education": ["education", "b.tech", "bachelor"],
        "Experience": ["experience", "internship"],
        "Projects": ["projects", "project"],
        "Skills": ["skills", "technical skills"],
        "Certifications": ["certification", "certifications"]
    }

    section_results = {}

    for section, keywords in sections.items():
        section_results[section] = any(
            keyword in resume_lower
            for keyword in keywords
        )

    sections_found = sum(section_results.values())
    structure_score = sections_found * 6

    # 3. Length Score (20)
    word_count = len(resume_text.split())

    if 250 <= word_count <= 900:
        length_score = 20
    elif 150 <= word_count < 250 or 900 < word_count <= 1200:
        length_score = 15
    else:
        length_score = 8

    # 4. Action Verb Score (10)
    action_verbs = [
        "developed",
        "built",
        "created",
        "analyzed",
        "implemented",
        "designed",
        "optimized",
        "trained",
        "automated",
        "managed"
    ]

    action_verb_count = sum(
        verb in resume_lower
        for verb in action_verbs
    )

    action_score = min(10, action_verb_count * 2)

    # Final score
    total_score = (
        skill_score
        + structure_score
        + length_score
        + action_score
    )

    # Display score
      # Display score
    st.subheader("📊 Resume Score")

    col1, col2, col3, col4, col5 = st.columns(5)

    col1.metric("Overall", f"{total_score}/100")
    col2.metric("Skills", f"{skill_score}/40")
    col3.metric("Structure", f"{structure_score}/30")
    col4.metric("Length", f"{length_score}/20")
    col5.metric("Action Verbs", f"{action_score}/10")

    st.progress(min(total_score, 100) / 100)

    if total_score >= 80:
        st.success("Excellent resume! 🚀")
    elif total_score >= 60:
        st.info("Good resume, but there is room for improvement. 👍")
    else:
        st.warning("Your resume needs improvement. 💡")
        
       # -----------------------------
    # JOB MATCHING
    # -----------------------------

    jobs = pd.DataFrame({
        "Job Title": [
            "Data Analyst",
            "Machine Learning Engineer",
            "Data Scientist",
            "Python Developer",
            "AI/ML Intern"
        ],

        "Skills": [
            "Python, SQL, Excel, Power BI, Pandas",
            "Python, Machine Learning, Scikit-learn, TensorFlow, Git",
            "Python, Pandas, NumPy, Machine Learning, Statistics",
            "Python, Flask, SQL, Git, REST API",
            "Python, Machine Learning, Pandas, NumPy, Git"
        ],

        "Job Description": [
            "Analyze business data, create reports and dashboards, clean datasets and find useful insights using Python, SQL, Pandas and Excel.",
            "Build and evaluate machine learning models, prepare datasets and develop AI solutions using Python, Machine Learning, Scikit-learn and TensorFlow.",
            "Analyze large datasets, build predictive models and communicate insights using Python, Pandas, NumPy, Machine Learning and Statistics.",
            "Develop backend applications and REST APIs using Python, Flask, SQL and Git.",
            "Assist with machine learning experiments, data preprocessing and model development using Python, Machine Learning, Pandas, NumPy and Git."
        ]
    })

    documents = [resume_text] + jobs["Job Description"].tolist()

    vectorizer = TfidfVectorizer(stop_words="english")
    tfidf_matrix = vectorizer.fit_transform(documents)

    similarity_scores = cosine_similarity(
        tfidf_matrix[0:1],
        tfidf_matrix[1:]
    )

    jobs["Match %"] = (
        similarity_scores[0] * 100
    ).round(2)

    jobs_sorted = jobs.sort_values(
        by="Match %",
        ascending=False
    )

    st.subheader("💼 Job Matches")

    # Get the best matching job
    top_job = jobs_sorted.iloc[0]

    st.success(
        f"⭐ Best Match: {top_job['Job Title']} | "
        f"Match Score: {top_job['Match %']}%"
    )

    st.write("### 📋 Job Match Ranking")

    for _, row in jobs_sorted.iterrows():
        st.write(
            f"**{row['Job Title']}** — "
            f"**{row['Match %']}% Match**"
        )

        st.progress(
            min(float(row["Match %"]) / 100, 1.0)
        )
       

    st.subheader("📊 Job Match Visualization")

    chart_data = jobs_sorted[["Job Title", "Match %"]].set_index("Job Title")

    st.bar_chart(chart_data)

    st.subheader("📊 Resume Score Breakdown")

    score_data = pd.DataFrame({
    "Category": ["Skills", "Structure", "Length", "Action Verbs"],
    "Score": [
        skill_score,
        structure_score,
        length_score,
        action_score
    ]
    }).set_index("Category")

    st.bar_chart(score_data)

        # -----------------------------
    # TOP JOB - MISSING SKILLS
    # -----------------------------

    st.subheader("🎯 Skills to Improve")

    top_job_title = top_job["Job Title"]

    required_skills = set(
        skill.strip().lower()
        for skill in top_job["Skills"].split(",")
    )

    resume_skills = set(found_skills)

    top_missing_skills = required_skills - resume_skills

    if top_missing_skills:

        st.write(
            f"For **{top_job_title}**, you should improve these skills:"
        )

        for skill in sorted(top_missing_skills):
            st.warning(f"❌ {skill.title()}")

    else:
        st.success(
            f"🎉 You already have all the listed skills for {top_job_title}!"
        )
        # -----------------------------
    # PERSONALIZED RECOMMENDATIONS
    # -----------------------------

    st.subheader("💡 Resume Improvement Recommendations")

    recommendations = []

    # Score-based recommendations
    if skill_score < 32:
        recommendations.append(
            "Add more relevant technical skills to improve your resume score."
        )

    if structure_score < 24:
        recommendations.append(
            "Improve your resume structure by clearly organizing Education, Experience, Projects and Skills."
        )

    if length_score < 20:
        recommendations.append(
            "Keep your resume concise and focused on relevant information."
        )

    if action_score < 8:
        recommendations.append(
            "Use stronger action verbs such as Developed, Built, Designed, Implemented and Analyzed."
        )

    # Job-based recommendation
    top_job = jobs_sorted.iloc[0]

    top_required_skills = set(
        skill.strip().lower()
        for skill in top_job["Skills"].split(",")
    )

    top_missing_skills = top_required_skills - resume_skills

    if top_missing_skills:
        recommendations.append(
            f"For your top match ({top_job['Job Title']}), "
            f"consider improving these skills: "
            f"{', '.join(sorted(top_missing_skills))}."
        )

    if not recommendations:
        recommendations.append(
            "Your resume looks strong! Keep adding relevant projects and measurable achievements."
        )

    for i, recommendation in enumerate(recommendations, 1):
        st.write(f"**{i}.** {recommendation}")

        # -----------------------------
    # DOWNLOAD REPORT
    # -----------------------------

    st.subheader("📥 Download Analysis Report")

    report = f"""
AI RESUME ANALYZER REPORT
=========================

RESUME SCORE
------------
Overall Score: {total_score}/100
Skills Score: {skill_score}/40
Structure Score: {structure_score}/30
Length Score: {length_score}/20
Action Verbs Score: {action_score}/10

DETECTED SKILLS
---------------
{", ".join(found_skills)}

BEST JOB MATCH
--------------
Job Title: {top_job["Job Title"]}
Match Score: {top_job["Match %"]}%

MISSING SKILLS
--------------
{", ".join(sorted(top_missing_skills)) if top_missing_skills else "None"}

RECOMMENDATIONS
---------------
"""

    for i, recommendation in enumerate(recommendations, 1):
        report += f"{i}. {recommendation}\n"

    st.download_button(
        label="📄 Download Resume Analysis",
        data=report,
        file_name="resume_analysis_report.txt",
        mime="text/plain"
    )