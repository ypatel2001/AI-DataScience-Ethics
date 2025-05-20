import json
import sys
import random
from collections import defaultdict

# Define the specific categories you want to use (modify this list as needed)
required_categories = [
    "Fashion & Apparel",
    "Technology & Digital",
    "Home & Lifestyle",
    "Travel & Transportation",
    "Media & Publishing"
]

def categorize_advertisers(advertisers):
    # Category definitions with keywords
    categories = {
        "Fashion & Apparel": [
            "Fossil", "Coach", "Vans", "UGG", "SHEIN", "kate spade", "Stuart Weitzman", 
            "SKECHERS", "Express", "Columbia", "Michael Kors", "Gymshark", "Levi's",
            "Lululemon", "Nike", "Adidas", "H&M", "Zara", "Gap", "Banana Republic",
            "Old Navy", "Forever 21", "Uniqlo", "Puma", "Under Armour", "Reebok",
            "Fashion", "Apparel", "Clothing", "Shoes"
        ],
        "Technology & Digital": [
            "Zoom", "Cloudflare", "Microsoft", "Meta", "WhatsApp", "EA - Electronic Arts", 
            "Motorola", "Ring", "Amazon", "Walmart.com", "Google", "Apple", "Samsung",
            "Dell", "HP", "Lenovo", "Intel", "AMD", "Nvidia", "Spotify", "Netflix",
            "Technology", "Software", "Hardware", "Electronics"
        ],
        "Home & Lifestyle": [
            "Le Creuset", "The Home Depot", "Wayfair", "Lands' End", "Bath & Body Works", 
            "Macy's", "Crate & Barrel", "Pottery Barn", "Williams Sonoma", "Bed Bath & Beyond",
            "IKEA", "HomeGoods", "Home", "Furniture", "Decor", "Kitchen"
        ],
        "Travel & Transportation": [
            "KAYAK", "Lyft", "Uber", "JetBlue", "T-Mobile", "American Airlines", 
            "Spirit Airlines", "Etihad Airways", "Air New Zealand", "Hertz", "Avis",
            "Expedia", "Travel", "Vacation", "Hotel", "Flight"
        ],
        "Media & Publishing": [
            "Harvard Business Review", "USA TODAY", "Penguin Random House", "Condé Nast", 
            "Vogue", "ELLE", "Harper's Bazaar", "Cosmopolitan", "Wired", "The New Yorker",
            "Time", "Forbes", "News", "Magazine", "Newspaper", "Media"
        ],
        "Food & Beverage": [
            "Starbucks", "McDonald's", "Subway", "Pizza Hut", "Domino's", "KFC",
            "Food", "Beverage", "Restaurant", "Cafe"
        ],
        "Health & Beauty": [
            "L'Oreal", "Maybelline", "Estee Lauder", "Clinique", "MAC", "NARS",
            "Health", "Beauty", "Skincare", "Makeup"
        ],
        "Other": []  # Catch-all for advertisers not matching the above
    }
    
    categorized = defaultdict(list)
    
    for advertiser in advertisers:
        matched = False
        for category, keywords in categories.items():
            if category == "Other":
                continue
            for keyword in keywords:
                if keyword.lower() in advertiser.lower():
                    categorized[category].append(advertiser)
                    matched = True
                    break
            if matched:
                break
        if not matched:
            categorized["Other"].append(advertiser)
    
    return categorized

def display_categories(categorized):
    # Display only categories with advertisers
    for category, advertisers in categorized.items():
        if advertisers:
            print(f"\n{category} ({len(advertisers)} advertisers):")
            print("=" * 50)
            for i, advertiser in enumerate(advertisers, 1):
                print(f"{i}. {advertiser}")

def main(file_path):
    try:
        # Load JSON data
        with open(file_path, 'r', encoding='utf-8') as file:
            data = json.load(file)
        
        # Extract advertisers from both sections
        all_advertisers = []
        for label in data.get('label_values', []):
            if label['label'] == 'A list uploaded or used by the advertiser':
                all_advertisers.extend([item['value'] for item in label['vec']])
        for label in data.get('label_values', []):
            if label['label'] == 'Interactions you may have had with the advertiser\'s website, app or store':
                all_advertisers.extend([item['value'] for item in label['vec']])
        
        # Remove duplicates
        seen = set()
        all_advertisers = [x for x in all_advertisers if not (x in seen or seen.add(x))]
        
        if not all_advertisers:
            print("No advertisers found in the specified sections.")
            return
        
        # Categorize all advertisers
        categorized = categorize_advertisers(all_advertisers)
        
        # Verify that each required category has at least 10 advertisers
        insufficient_categories = [cat for cat in required_categories if len(categorized.get(cat, [])) < 10]
        if insufficient_categories:
            print("The following required categories have fewer than 10 advertisers:")
            for cat in insufficient_categories:
                print(f"- {cat}: {len(categorized.get(cat, []))} advertisers")
            print("Please adjust the 'required_categories' list or ensure sufficient data.")
            return
        
        # Select 10 advertisers from each required category
        sample = []
        for cat in required_categories:
            advs = categorized[cat]
            selected_advs = random.sample(advs, 10)
            sample.extend(selected_advs)
            # Remove selected advertisers to avoid reuse
            categorized[cat] = [adv for adv in advs if adv not in selected_advs]
        
        # Create remaining pool from only the required categories
        remaining_pool = []
        for cat in required_categories:
            remaining_pool.extend(categorized[cat])
        
        # Add additional advertisers to reach 100-150 total, if possible
        if remaining_pool:
            additional_count = min(len(remaining_pool), random.randint(50, 100))
            additional_sample = random.sample(remaining_pool, additional_count)
            sample.extend(additional_sample)
        
        # Categorize the final sample
        sample_categorized = categorize_advertisers(sample)
        
        # Display results
        print(f"\nTotal advertisers in sample: {len(sample)}")
        display_categories(sample_categorized)
        
    except Exception as e:
        print(f"Error: {str(e)}")

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python categorize_advertisers.py <json_file_path>")
        sys.exit(1)
    file_path = sys.argv[1]
    main(file_path)