package com.android.victor.model

data class MessageModel(
    var messageText: String = "",
    var type: String = "victor",
    var push: Boolean = false,
    var messageId: String = "",
    var textLength: String = "",
)
