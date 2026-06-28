# Golden evaluation set for the Tesla 2022 10-K (fiscal year ended Dec 31, 2022)
# All figures verified directly from the consolidated financial statements.
# "expected" holds the key fact the answer must convey. The LLM judge checks
# semantic correctness, so exact formatting in the answer doesn't matter.

eval_set = [
    # ---------- Exact-figure questions (test keyword/BM25 retrieval) ----------
    {
        "question": "What was Tesla's total revenue in 2022?",
        "expected": "$81,462 million (about $81.5 billion)",
    },
    {
        "question": "What was Tesla's total automotive revenue in 2022?",
        "expected": "$71,462 million",
    },
    {
        "question": "What was automotive sales revenue in 2022?",
        "expected": "$67,210 million",
    },
    {
        "question": "What was automotive regulatory credits revenue in 2022?",
        "expected": "$1,776 million",
    },
    {
        "question": "What was automotive leasing revenue in 2022?",
        "expected": "$2,476 million",
    },
    {
        "question": "What was services and other revenue in 2022?",
        "expected": "$6,091 million",
    },
    {
        "question": "What was Tesla's gross profit in 2022?",
        "expected": "$20,853 million",
    },
    {
        "question": "What was income from operations in 2022?",
        "expected": "$13,656 million",
    },
    {
        "question": "How much did Tesla spend on research and development in 2022?",
        "expected": "$3,075 million",
    },
    {
        "question": "What was selling, general and administrative expense in 2022?",
        "expected": "$3,946 million",
    },
    {
        "question": "What was net income attributable to common stockholders in 2022?",
        "expected": "$12,556 million (about $12.5 billion)",
    },
    {
        "question": "What was basic net income per share in 2022?",
        "expected": "$4.02",
    },
    {
        "question": "What was the cost of automotive sales revenue in 2022?",
        "expected": "$49,599 million",
    },
    {
        "question": "What was total automotive gross margin in 2022?",
        "expected": "28.5% (down from 29.3% in 2021)",
    },
    {
        "question": "How much FSD revenue was recognized in Q4 2022 in North America?",
        "expected": "$324 million",
    },
    {
        "question": "By how many units did Model 3 and Model Y deliveries increase in 2022 versus 2021?",
        "expected": "347,024 additional deliveries",
    },

    # ---------- Open-ended questions (test semantic retrieval + reasoning) ----------
    {
        "question": "What is Tesla's stated mission?",
        "expected": "To accelerate the world's transition to sustainable energy.",
    },
    {
        "question": "What challenges does Tesla face in scaling production at new factories?",
        "expected": "Construction timeline/cost overruns, production ramp difficulties, regulatory compliance, supply chain and battery cell constraints, and labor shortages.",
    },
    {
        "question": "How does Tesla approach battery safety in its vehicles?",
        "expected": "Battery packs are designed to passively contain a single cell's energy release without spreading to neighboring cells, though fire/failure risk remains, especially in high-speed crashes.",
    },
    {
        "question": "How does Tesla recognize revenue from automotive sales?",
        "expected": "Revenue is recognized upon delivery, when control of the vehicle transfers to the customer, in line with ASC 606.",
    },
    {
        "question": "What are Tesla's main manufacturing locations?",
        "expected": "Fremont Factory (California), Gigafactory Nevada, Gigafactory Shanghai, Gigafactory Berlin-Brandenburg, and Gigafactory Texas.",
    },
    {
        "question": "Why did automotive sales revenue increase in 2022?",
        "expected": "Higher Model 3/Model Y and Model S/X deliveries from production ramp at Shanghai, Fremont, and new Berlin and Texas factories, at a higher average selling price, partly offset by a stronger US dollar.",
    },
    {
        "question": "What does Tesla's services and other revenue consist of?",
        "expected": "Non-warranty after-sales service and parts, paid Supercharging, used vehicle sales, retail merchandise, and vehicle insurance.",
    },
    {
        "question": "Why did SG&A expense decrease in 2022?",
        "expected": "Mainly a decrease in stock-based compensation expense, largely from lower expense on the 2018 CEO Performance Award, partly offset by headcount growth.",
    },
    {
        "question": "What are some key risks Tesla identifies related to its business?",
        "expected": "Risks include production ramp and supply chain challenges, battery cell supply and safety, product liability, regulatory compliance, and demand generation, among others.",
    },
]
