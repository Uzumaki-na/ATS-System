"""
Build skill database JSON from SkillNer SKILL_DB + curated aliases.

Usage: .venv/Scripts/python scripts/build_skill_db.py
Output: data/skill_db.json (~32K skills + aliases)
"""

import json
import os
import sys
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def extract_skillner_db() -> list:
    """Extract all skills from SkillNer's SKILL_DB."""
    from skillNer.general_params import SKILL_DB

    skills = []
    seen_names = set()

    for skill_id, entry in SKILL_DB.items():
        name = entry.get("skill_name", "").strip()
        if not name or name.lower() in seen_names:
            continue
        seen_names.add(name.lower())

        skill_type = entry.get("skill_type", "Hard Skill")

        # Map SkillNer types to our categories
        if skill_type == "Soft Skill":
            category = "soft_skill"
        elif "cert" in name.lower() or skill_type == "Certification":
            category = "certification"
        else:
            category = "hard_skill"

        # Build aliases: the skill name itself + any high_surface_form if available
        aliases = [name.lower()]
        hsf = entry.get("high_surface_form", "")
        if hsf and hsf.lower() not in aliases:
            aliases.append(hsf.lower())

        skills.append({
            "name": name,
            "category": category,
            "aliases": aliases,
        })

    return skills


def add_curated_skills() -> list:
    """Hand-curated technical skills and abbreviations SkillNer misses."""
    return [
        # ── Programming Languages (with abbreviations) ──
        {"name": "Python", "category": "programming_language", "aliases": ["python", "python3", "py", "python programming"]},
        {"name": "JavaScript", "category": "programming_language", "aliases": ["javascript", "js", "ecmascript", "es6", "es2015"]},
        {"name": "TypeScript", "category": "programming_language", "aliases": ["typescript", "ts"]},
        {"name": "C++", "category": "programming_language", "aliases": ["c++", "cpp", "c plus plus", "cxx"]},
        {"name": "C#", "category": "programming_language", "aliases": ["c#", "c sharp", "csharp"]},
        {"name": "Go", "category": "programming_language", "aliases": ["golang", "go language"]},
        {"name": "Rust", "category": "programming_language", "aliases": ["rust", "rustlang"]},
        {"name": "Scala", "category": "programming_language", "aliases": ["scala"]},
        {"name": "Kotlin", "category": "programming_language", "aliases": ["kotlin"]},
        {"name": "Swift", "category": "programming_language", "aliases": ["swift"]},
        {"name": "Dart", "category": "programming_language", "aliases": ["dart", "flutter"]},

        # ── ML / Data Science ──
        {"name": "Machine Learning", "category": "ml_ds", "aliases": ["machine learning", "ml"]},
        {"name": "Deep Learning", "category": "ml_ds", "aliases": ["deep learning", "dl"]},
        {"name": "Natural Language Processing", "category": "ml_ds", "aliases": ["natural language processing", "nlp"]},
        {"name": "Computer Vision", "category": "ml_ds", "aliases": ["computer vision", "cv", "machine vision"]},
        {"name": "Data Science", "category": "ml_ds", "aliases": ["data science"]},
        {"name": "Data Engineering", "category": "ml_ds", "aliases": ["data engineering"]},
        {"name": "Data Analytics", "category": "ml_ds", "aliases": ["data analytics", "data analysis", "analytics"]},
        {"name": "Business Intelligence", "category": "ml_ds", "aliases": ["business intelligence", "bi"]},
        {"name": "Artificial Intelligence", "category": "ml_ds", "aliases": ["artificial intelligence", "ai"]},
        {"name": "Generative AI", "category": "ml_ds", "aliases": ["generative ai", "gen ai", "genai"]},
        {"name": "Large Language Models", "category": "ml_ds", "aliases": ["large language models", "llm", "llms"]},
        {"name": "RAG", "category": "ml_ds", "aliases": ["retrieval augmented generation", "rag"]},
        {"name": "Recommender Systems", "category": "ml_ds", "aliases": ["recommender systems", "recommender system", "recsys"]},
        {"name": "Time Series", "category": "ml_ds", "aliases": ["time series", "time series analysis", "time series forecasting"]},

        # ── Cloud / DevOps ──
        {"name": "Kubernetes", "category": "devops", "aliases": ["kubernetes", "k8s", "kube"]},
        {"name": "Docker", "category": "devops", "aliases": ["docker"]},
        {"name": "Terraform", "category": "devops", "aliases": ["terraform", "tf"]},
        {"name": "Ansible", "category": "devops", "aliases": ["ansible"]},
        {"name": "GitHub Actions", "category": "devops", "aliases": ["github actions", "gh actions", "github action"]},
        {"name": "GitLab CI", "category": "devops", "aliases": ["gitlab ci", "gitlab-ci"]},
        {"name": "Prometheus", "category": "devops", "aliases": ["prometheus"]},
        {"name": "Grafana", "category": "devops", "aliases": ["grafana"]},
        {"name": "Helm", "category": "devops", "aliases": ["helm", "helm charts", "helm chart"]},
        {"name": "Istio", "category": "devops", "aliases": ["istio", "istio service mesh"]},
        {"name": "Service Mesh", "category": "devops", "aliases": ["service mesh"]},
        {"name": "Microservices", "category": "devops", "aliases": ["microservices", "micro service", "micro service architecture"]},
        {"name": "Serverless", "category": "devops", "aliases": ["serverless", "server less"]},
        {"name": "SRE", "category": "devops", "aliases": ["site reliability engineering", "sre"]},
        {"name": "MLOps", "category": "devops", "aliases": ["mlops", "ml ops"]},
        {"name": "DevOps", "category": "devops", "aliases": ["devops"]},
        {"name": "CI/CD", "category": "devops", "aliases": ["ci/cd", "ci cd", "continuous integration", "continuous deployment", "continuous delivery"]},

        # ── Web / Full-Stack ──
        {"name": "Node.js", "category": "framework", "aliases": ["node.js", "nodejs", "node", "node js"]},
        {"name": "React", "category": "framework", "aliases": ["react", "react.js", "reactjs", "react js"]},
        {"name": "Angular", "category": "framework", "aliases": ["angular", "angular.js", "angularjs", "angular 2+"]},
        {"name": "Vue.js", "category": "framework", "aliases": ["vue", "vue.js", "vuejs"]},
        {"name": "Next.js", "category": "framework", "aliases": ["next.js", "nextjs", "next"]},
        {"name": "Nuxt.js", "category": "framework", "aliases": ["nuxt", "nuxt.js", "nuxtjs"]},
        {"name": "Svelte", "category": "framework", "aliases": ["svelte", "sveltekit"]},
        {"name": "Spring Boot", "category": "framework", "aliases": ["spring boot", "springboot"]},
        {"name": "Spring Framework", "category": "framework", "aliases": ["spring", "spring framework"]},
        {"name": ".NET", "category": "framework", "aliases": [".net", "dotnet", "dot net"]},
        {"name": "ASP.NET", "category": "framework", "aliases": ["asp.net", "aspnet", "asp dot net"]},
        {"name": "GraphQL", "category": "tool", "aliases": ["graphql", "gql"]},
        {"name": "REST API", "category": "tool", "aliases": ["rest api", "restful api", "rest", "restful", "rest apis"]},
        {"name": "gRPC", "category": "tool", "aliases": ["grpc", "g rpc"]},

        # ── Databases ──
        {"name": "SQL", "category": "database", "aliases": ["sql"]},
        {"name": "NoSQL", "category": "database", "aliases": ["nosql", "no sql"]},
        {"name": "PostgreSQL", "category": "database", "aliases": ["postgresql", "postgres", "psql"]},
        {"name": "MongoDB", "category": "database", "aliases": ["mongodb", "mongo"]},
        {"name": "Redis", "category": "database", "aliases": ["redis"]},
        {"name": "Elasticsearch", "category": "database", "aliases": ["elasticsearch", "es", "elastic"]},
        {"name": "BigQuery", "category": "database", "aliases": ["bigquery", "big query"]},
        {"name": "Snowflake", "category": "database", "aliases": ["snowflake"]},
        {"name": "Redshift", "category": "database", "aliases": ["redshift", "amazon redshift"]},
        {"name": "Databricks", "category": "database", "aliases": ["databricks"]},

        # ── Tools ──
        {"name": "Git", "category": "tool", "aliases": ["git", "git vcs"]},
        {"name": "Jupyter", "category": "tool", "aliases": ["jupyter", "jupyter notebook", "jupyter lab"]},
        {"name": "VS Code", "category": "tool", "aliases": ["vs code", "vscode", "visual studio code"]},

        # ── Soft Skills ──
        {"name": "Problem Solving", "category": "soft_skill", "aliases": ["problem solving", "problem-solving"]},
        {"name": "Critical Thinking", "category": "soft_skill", "aliases": ["critical thinking", "critical-thinking"]},
        {"name": "Agile", "category": "soft_skill", "aliases": ["agile", "agile methodology"]},
        {"name": "Scrum", "category": "soft_skill", "aliases": ["scrum", "scrum master"]},
        {"name": "Kanban", "category": "soft_skill", "aliases": ["kanban"]},
        {"name": "Cross-functional Collaboration", "category": "soft_skill", "aliases": ["cross functional", "cross-functional", "cross team"]},
        {"name": "Stakeholder Management", "category": "soft_skill", "aliases": ["stakeholder management", "stakeholder"]},
        {"name": "Mentoring", "category": "soft_skill", "aliases": ["mentoring", "mentorship"]},
        {"name": "Presentation", "category": "soft_skill", "aliases": ["presentation", "presentation skills"]},
    ]


def deduplicate(skills: list) -> list:
    """Remove duplicates: if two entries have the same alias, keep the first."""
    seen_aliases = set()
    deduped = []
    for skill in skills:
        unique_aliases = []
        for alias in skill["aliases"]:
            if alias not in seen_aliases:
                seen_aliases.add(alias)
                unique_aliases.append(alias)
        if unique_aliases:
            skill["aliases"] = unique_aliases
            deduped.append(skill)
    return deduped


if __name__ == "__main__":
    print("Extracting SkillNer skills...")
    skillner_skills = extract_skillner_db()
    print(f"  Got {len(skillner_skills)} unique skills from SkillNer")

    curated = add_curated_skills()
    print(f"  Got {len(curated)} curated skills")

    combined = deduplicate(skillner_skills + curated)
    print(f"  Combined: {len(combined)} unique skills after dedup")

    # Stats
    from collections import Counter
    cats = Counter(s["category"] for s in combined)
    print(f"\nCategory breakdown:")
    for cat, count in cats.most_common():
        print(f"  {cat}: {count}")

    # Save
    output_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "data",
        "skill_db.json",
    )
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump({"skills": combined}, f, indent=2)

    print(f"\nSaved to {output_path}")
    alias_count = sum(len(s["aliases"]) for s in combined)
    print(f"Total aliases: {alias_count}")
