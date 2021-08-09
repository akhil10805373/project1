from flask import jsonify

import globals


def heartbeat():
    if len(globals.mean_values) < 200 or (len(globals.latest_bpm) <= 0):
        return "Not Enough Data!"
    else:
        globals.latest_bpm = globals.latest_bpm[-10:]
        result = {"bpm": [globals.latest_bpm[-1]]}
        return jsonify(result)
