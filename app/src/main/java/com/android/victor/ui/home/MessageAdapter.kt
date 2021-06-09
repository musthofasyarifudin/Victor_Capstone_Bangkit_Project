package com.android.victor.ui.home

import android.util.Log
import android.view.Gravity
import android.view.LayoutInflater
import android.view.View
import android.view.View.GONE
import android.view.ViewGroup
import android.widget.LinearLayout
import android.widget.RelativeLayout
import androidx.appcompat.widget.AppCompatTextView
import androidx.recyclerview.widget.RecyclerView
import com.android.victor.R
import com.android.victor.model.MessageModel
import com.google.android.material.card.MaterialCardView


class MessageAdapter(var message: ArrayList<MessageModel>) :
    RecyclerView.Adapter<MessageAdapter.MessageViewHolder>() {

    var listener: OnItemClick? = null

    fun countItemMessage(): Int {
        return this.message.size
    }

    fun addSingleMessage(message: MessageModel){
        this.message.add(message)
        notifyDataSetChanged()
    }

    fun addItemMessage(message: ArrayList<MessageModel>) {
        this.message.addAll(message)
        notifyDataSetChanged()
    }

    override fun onBindViewHolder(holder: MessageViewHolder, position: Int) {
        holder.bind(message[position])
        holder.cvMessageVictor.setOnClickListener{
            listener?.onMessageClick(message[position])
        }
    }

    override fun getItemCount(): Int = message.size

    override fun onCreateViewHolder(
        parent: ViewGroup,
        viewType: Int
    ): MessageViewHolder {
        val view: View = LayoutInflater.from(parent.context)
            .inflate(R.layout.item_message, parent, false)
        return MessageViewHolder(view)
    }

    class MessageViewHolder(itemView: View) : RecyclerView.ViewHolder(itemView) {
        var tvMessageVictor: AppCompatTextView = itemView.findViewById(R.id.tvMessageVictor)
        var cvMessageVictor: MaterialCardView = itemView.findViewById(R.id.cvMessageVictor)
        var params = RelativeLayout.LayoutParams(RelativeLayout.LayoutParams.MATCH_PARENT, RelativeLayout.LayoutParams.WRAP_CONTENT)

        fun bind(message: MessageModel){
            if (message.type == "victor" && message.textLength == "Long"){
                params.removeRule(RelativeLayout.END_OF)
                params.marginEnd = 100
                params.addRule(RelativeLayout.ALIGN_PARENT_END)
                params.addRule(RelativeLayout.ALIGN_PARENT_START)
                tvMessageVictor.textAlignment = AppCompatTextView.TEXT_ALIGNMENT_VIEW_START
            } else if (message.type == "victor"){
                params.removeRule(RelativeLayout.END_OF)
                params.removeRule(RelativeLayout.ALIGN_PARENT_END)
                params.marginEnd = 0
                params.addRule(RelativeLayout.START_OF, R.id.view)
                params.addRule(RelativeLayout.ALIGN_PARENT_START)
                tvMessageVictor.textAlignment = AppCompatTextView.TEXT_ALIGNMENT_VIEW_START
            } else {
                params.removeRule(RelativeLayout.START_OF)
                params.removeRule(RelativeLayout.ALIGN_PARENT_START)
                params.marginEnd = 0
                params.addRule(RelativeLayout.END_OF, R.id.view)
                params.addRule(RelativeLayout.ALIGN_PARENT_END)
                tvMessageVictor.textAlignment = AppCompatTextView.TEXT_ALIGNMENT_VIEW_END
            }
            cvMessageVictor.layoutParams = params
            tvMessageVictor.text = message.messageText
        }
    }

    interface OnItemClick{
        fun onMessageClick(message: MessageModel)
    }
}