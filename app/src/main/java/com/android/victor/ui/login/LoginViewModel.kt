package com.android.victor.ui.login

import androidx.lifecycle.MutableLiveData
import androidx.lifecycle.ViewModel
import com.android.victor.model.User
import com.android.victor.utils.Utils.FirestoreKeys.COLLECTION_USER
import com.android.victor.utils.Utils.FirestoreKeys.FIELD_EMAIL
import com.android.victor.utils.Utils.FirestoreKeys.FIELD_NAME
import com.android.victor.utils.Utils.FirestoreKeys.FIELD_PHONE
import com.google.android.gms.tasks.Task
import com.google.firebase.auth.AuthResult
import com.google.firebase.auth.FirebaseAuth
import com.google.firebase.auth.FirebaseUser
import com.google.firebase.firestore.FirebaseFirestore

class LoginViewModel: ViewModel() {

    private lateinit var auth: FirebaseAuth
    private lateinit var firestore: FirebaseFirestore
    private lateinit var firebaseUser: FirebaseUser

    var user = MutableLiveData<User>()
    var loading = MutableLiveData<Boolean>()
    var errorMessage = MutableLiveData<String>()
    var showMessage = MutableLiveData<Boolean>()
    var successSignIn = MutableLiveData<Boolean>()

    fun checkEmailInUser(email: String, password: String) {
        defaultValue()
        auth = FirebaseAuth.getInstance()
        firestore = FirebaseFirestore.getInstance()
        loading.value = true
        firestore.collection(COLLECTION_USER)
            .whereEqualTo(FIELD_EMAIL, email)
            .get()
            .addOnCompleteListener { task ->
                if (task.isSuccessful){
                    if (task.result?.isEmpty == true){
                        loading.value = false
                        errorMessage.value = "Email is not registered yet, please register first"
                        showMessage.value = true
                    } else {
                        val data = task.result?.documents?.get(0)
                        val user = User(
                            data?.getString(FIELD_NAME)!!,
                            data.getString(FIELD_EMAIL)!!,
                            data.getString(FIELD_PHONE)!!
                        )
                        requestSignIn(user, email, password)
                    }
                } else {
                    loading.value = false
                    task.exception?.message?.let { message -> errorMessage.value = message }
                    showMessage.value = true
                }
            }
    }

    private fun requestSignIn(userLogin: Any, email: String, password: String) {
        auth.signInWithEmailAndPassword(email, password)
            .addOnCompleteListener { task: Task<AuthResult?> ->
                if (task.isSuccessful) {
                    firebaseUser = auth.currentUser!!
                    if (firebaseUser.isEmailVerified) {
                        loading.value = false
                        successSignIn.value = true
                    } else {
                        auth.signOut()
                        loading.value = false
                        errorMessage.value = "Your email has not been verified"
                        showMessage.value = true
                    }
                } else {
                    loading.value = false
                    task.exception?.message?.let { message -> errorMessage.value = message }
                    showMessage.value = true
                }
            }
    }

    private fun defaultValue() {
        successSignIn.value = false
        loading.value = false
        showMessage.value = false
        errorMessage.value = ""
        user.value = User()
    }
}