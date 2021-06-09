package com.android.victor.ui.intro

import android.view.LayoutInflater
import android.view.View
import android.view.ViewGroup
import android.widget.ImageView
import android.widget.TextView
import com.android.victor.R
import com.android.victor.model.SlideModel
import com.bumptech.glide.Glide
import com.smarteist.autoimageslider.SliderViewAdapter

class IntroAdapter(private var items: ArrayList<SlideModel>): SliderViewAdapter<IntroAdapter.SliderVH>() {

    fun updateImage(item: ArrayList<SlideModel>) {
        items.clear()
        items.addAll(item)
        notifyDataSetChanged()
    }

    override fun onCreateViewHolder(parent: ViewGroup?) = SliderVH(
        LayoutInflater.from(parent?.context).inflate(R.layout.item_slider_view, parent, false)
    )

    override fun getCount() = items.size

    override fun onBindViewHolder(viewHolder: SliderVH, position: Int) {
        viewHolder.bind(items[position])
    }

    class SliderVH(itemView: View) : SliderViewAdapter.ViewHolder(itemView) {
        private val image = itemView.findViewById<ImageView>(R.id.iv_auto_image_slider)
        private val title = itemView.findViewById<TextView>(R.id.tv_title)
        private val description = itemView.findViewById<TextView>(R.id.tv_description)
        fun bind(item: SlideModel) {
            title.text = item.title
            description.text = item.description
            Glide.with(itemView.context).load(item.image).into(image)
        }
    }
}