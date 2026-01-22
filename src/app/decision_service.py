import sys
import os
import json
from commons.FileUtils import FileUtils
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq
import shutil
from commons.config_reader import config
from commons.constants import Constants as Co

if __name__ == "__main__":
    root_folder = ""
    output_root = root_folder+"src/model_output"
    model_name = config[Co.LLM][Co.MODEL]
    bills_map = {}  # key: "emp_id_emp_name", value: list of bills
    bills = []

    # Scan all categories under output (meal, commute, etc.)
    for category in os.listdir(output_root):
        category_path = os.path.join(output_root, category)
        if not os.path.isdir(category_path) or category == "policy":
            continue  # skip non-folders and policy directory
        category_path=category_path+"/"+model_name

        for fname in os.listdir(category_path):
            if os.path.isdir(fname):
                continue
            full_path = os.path.join(category_path, fname)
            try:
                file_bills = FileUtils.load_json_from_file(full_path)
                if not isinstance(file_bills, list):
                    file_bills = [file_bills]

                for b in file_bills:
                    emp_id = b.get("emp_id", "")
                    emp_name = b.get("emp_name", "")
                    key = f"{emp_id}_{emp_name}"

                    # Attach category (from folder name, e.g., meal or commute)
                    if "category" not in b or not b["category"]:
                        b["category"] = category

                    bills_map.setdefault(key, []).append(b)

            except Exception as e:
                print(f"⚠️ Failed to load {category}/{fname}: {e}")

    #print(bills_map)
    # Flatten bills if you still need a list elsewhere
    bills = [bill for bills_list in bills_map.values() for bill in bills_list]

    # Load policy JSON from output
    policy = FileUtils.load_json_from_file(
        root_folder+"src/model_output/policy/"+model_name+"/policy.json"
    )

    if not bills:
        print("❌ No bills found in  output files.")
        sys.exit(1)

    # Prepare groups_data for the LLM
    groups_data = []
    save_data = []

    for key, emp_bills in bills_map.items():
        emp_id, emp_name = key.split("_", 1)

        # 🔹 Step 1: Group employee bills by category
        category_groups = {}
        for b in emp_bills:
            cat = b.get("category", "unknown")
            category_groups.setdefault(cat, []).append(b)

        # 🔹 Step 2: Process each category separately
        for category, cat_bills in category_groups.items():
            valid_for_group = [b for b in cat_bills if b.get("validation", {}).get("is_valid")]
            invalid_for_group = [b for b in cat_bills if not b.get("validation", {}).get("is_valid")]

            # Construct data groups for LLM according to category
            daily_totals = {}
            for b in valid_for_group:
                invoice_date = b.get("date")
                if invoice_date not in daily_totals:
                    daily_totals[invoice_date] = 0
                daily_totals[invoice_date] += float(b.get("amount", 0) or 0)

            if category == "meal" and daily_totals:
                # 🍱 One group per date for meal bills
                for date, total in daily_totals.items():
                    groups_data.append({
                        "employee_id": emp_id,
                        "employee_name": emp_name,
                        "category": category,
                        "date": date,
                        "valid_bills": [
                            b.get("id")
                            for b in valid_for_group if b.get("date") == date
                        ],
                        "invalid_bills": [
                            b.get("id")
                            for b in invalid_for_group if b.get("date") == date
                        ],
                        "daily_total": total,
                        "monthly_total": None
                    })
            else:
                # 🚗 For commute/fuel/other categories — keep one record per month
                groups_data.append({
                    "employee_id": emp_id,
                    "employee_name": emp_name,
                    "category": category,
                    "date": None,
                    "valid_bills": [b.get("id") for b in valid_for_group],
                    "invalid_bills": [b.get("id") for b in invalid_for_group],
                    "daily_total": None,
                    "monthly_total": sum(float(b.get("amount", 0) or 0)
                                         for b in valid_for_group)
                })

            # Save metadata for copying files later
            save_data.append({
                "employee_id": emp_id,
                "employee_name": emp_name,
                "category": category,
                "valid_files": [b.get("filename") for b in valid_for_group],
                "invalid_files": [b.get("filename") for b in invalid_for_group]
            })

    # Debug print
    print(f"🗂 Prepared {groups_data} groups for LLM processing.")

    # Construct user prompt using all groups
    user_prompt = json.dumps({
        "policy": policy,
        "groups": groups_data
    }, indent=2)

    # Load and append system prompt
    system_prompt = FileUtils.load_text_file(
        root_folder+"src/prompt/system_prompt_decision.txt"
    )

    prompt = ChatPromptTemplate.from_messages([
        ("system", "{system_prompt}"),
        ("human", "{user_prompt}")
    ])

    llm = ChatGroq(
        model=model_name,
        temperature=config[Co.LLM][Co.TEMPERATURE]
    )

    parser = StrOutputParser()

    chain = prompt | llm | parser


    output = chain.invoke({
        "system_prompt": system_prompt,
        "user_prompt": user_prompt
    })


    print("\n📄 All Decisions Output:")
    print(output)

    valid_base_dir = output_root+"/{category}/"+model_name+"/valid_bills"
    invalid_base_dir = output_root+"/{category}/"+model_name+"/invalid_bills"
    src_resources_root = root_folder+"resources/{category}"

    for emp in save_data:
        emp_id = emp.get("employee_id")
        emp_name = emp.get("employee_name")
        category = emp.get("category")
        if category == "cab":
            category = "commute"

        emp_valid_dir = os.path.join(valid_base_dir.replace("{category}",category), f"{emp_id}_{emp_name}")
        emp_invalid_dir = os.path.join(invalid_base_dir.replace("{category}",category), f"{emp_id}_{emp_name}")
        os.makedirs(emp_valid_dir, exist_ok=True)
        os.makedirs(emp_invalid_dir, exist_ok=True)

        valid_files = emp.get("valid_files", [])
        invalid_files = emp.get("invalid_files", [])

        # Determine the resources source folder for the employee
        resources_src_dir = None
        for folder_name in os.listdir(src_resources_root.replace("{category}",category)):
            if folder_name.startswith(emp_id):
                resources_src_dir = os.path.join(src_resources_root.replace("{category}",category), folder_name)
                break

        if not resources_src_dir:
            print(f"⚠️ No meal source found for {emp_id}_{emp_name}")
            continue

        # Copy valid files
        for fname in os.listdir(resources_src_dir):
            for vf in valid_files:
                if vf and vf in fname:
                    src_path = os.path.join(resources_src_dir, fname)
                    dest_path = os.path.join(emp_valid_dir, fname)
                    shutil.copy(src_path, dest_path)

        # Copy invalid files
        for fname in os.listdir(resources_src_dir):
            for inf in invalid_files:
                if inf and inf in fname:
                    src_path = os.path.join(resources_src_dir, fname)
                    dest_path = os.path.join(emp_invalid_dir, fname)
                    shutil.copy(src_path, dest_path)

        print(f"✅ Copied {category} files for {emp_id}_{emp_name}: {len(valid_files)} valid, {len(invalid_files)} invalid.")
