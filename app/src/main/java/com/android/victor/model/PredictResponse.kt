package com.android.victor.model

class PredictResponse(
    var resp: Int,
    var probability: String,
    var prediction: String,
    var desc: String,
    var precautions: ArrayList<String>
)