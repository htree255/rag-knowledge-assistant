# eatsy_kb_dataset.py

# A dataset with 20 example entries
data = [
{"id": f"faq_{i:03}", "text": text, "category": category, "source": source, "restaurant_id": restaurant_id, "date_created": "2025-09-12"}
for i, (text, category, source, restaurant_id) in enumerate([
("How can I track my order? Go to 'My Orders' in your profile.", "tracking", "FAQ", None),
("Can I cancel an order? Orders can be canceled within 5 minutes.", "cancellation", "FAQ", None),
("Refunds are processed within 100 business days.", "refund", "FAQ", None),
("Vegan Burger: lettuce, tomato, vegan mayo.", "menu", "Restaurant Menu", "rest_101"),
("Green Smoothie: kale, spinach, apple, almond milk.", "menu", "Restaurant Menu", "rest_102"),
("If your package arrived damaged, contact support with a photo.", "complaint", "Support Transcript", None),
("We deliver within 30 minutes in city center.", "delivery", "FAQ", None),
("Payment methods: credit card, PayPal, Apple Pay.", "payment", "FAQ", None),
("All our restaurants are certified vegan/vegetarian.", "policy", "FAQ", None),
("Gluten-free options are available.", "menu", "Restaurant Menu", "rest_103"),
("Late delivery? Contact support to get an update.", "delivery", "Support Transcript", None),
("Happy hour discounts available 5-7pm.", "promo", "Restaurant Menu", "rest_104"),
("We prioritize eco-friendly packaging.", "policy", "FAQ", None),
("Order history is saved for 6 months.", "account", "FAQ", None),
("Vegan sushi rolls available.", "menu", "Restaurant Menu", "rest_105"),
("Your feedback is valuable to us.", "feedback", "Support Transcript", None),
("We support allergy-friendly meals.", "menu", "Restaurant Menu", "rest_106"),
("Order customization allowed for dietary restrictions.", "menu", "Restaurant Menu", "rest_107"),
("Contact support via chat, email, or phone.", "support", "FAQ", None),
("Refer a friend and get discounts.", "promo", "FAQ", None)
], start=1)
]

# Save as CSV for ingestion
# kb_df.to_csv("eatsy_kb.csv", index=False)