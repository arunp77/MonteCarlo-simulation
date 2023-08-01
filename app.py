from flask import Flask, request, jsonify
import random

app = Flask(__name__)

@app.route('/simulate', methods=['POST'])
def simulate_roll():
    num_rolls = 1000000
    rolls = []
    sum_values = 0

    for _ in range(num_rolls):
        roll = random.randint(1, 6)
        rolls.append(roll)
        sum_values += roll

    average = sum_values / num_rolls

    return jsonify({"average": average})

if __name__ == '__main__':
    app.run(debug=True)
