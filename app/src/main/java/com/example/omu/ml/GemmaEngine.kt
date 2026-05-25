package com.example.omu.ml

import android.content.Context
import android.util.Log
import com.google.ai.edge.litertlm.Backend
import com.google.ai.edge.litertlm.Content
import com.google.ai.edge.litertlm.Contents
import com.google.ai.edge.litertlm.Engine
import com.google.ai.edge.litertlm.EngineConfig
import com.google.ai.edge.litertlm.Message
import com.google.ai.edge.litertlm.MessageCallback
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.flow.Flow
import kotlinx.coroutines.flow.map
import kotlinx.coroutines.flow.onCompletion
import kotlinx.coroutines.withContext
import java.io.File


class GemmaEngine(private val context: Context, private val modelPath: String) {
    private var engine: Engine? = null
    private var customConfig: EngineConfig? = null

    suspend fun initialize() = withContext(Dispatchers.IO) {
        val config = customConfig ?: EngineConfig(
            modelPath = modelPath,
            backend = Backend.GPU(),
            cacheDir = context.cacheDir.path
        )
        val newEngine = Engine(config)
        newEngine.initialize()
        engine = newEngine
    }

    fun generateResponseAsync(contents: Contents, onChunk: (Message) -> Unit) {
        val activeEngine = engine ?: return
        val conversation = activeEngine.createConversation()
        
        conversation.sendMessageAsync(contents, object : MessageCallback {
            override fun onMessage(message: Message) {
                onChunk(message)
            }

            override fun onDone() {
                conversation.close()
            }

            override fun onError(throwable: Throwable) {
                conversation.close()
            }
        })
    }

    fun generateResponse(contents: Contents): Flow<String>? {
        val activeEngine = engine ?: return null
        val conversation = activeEngine.createConversation()
        return conversation.sendMessageAsync(contents)
            .map { it.toString() }
            .onCompletion { conversation.close() }
    }

    fun generateResponse(prompt: String): Flow<String>? {
        return generateResponse(Contents.of(Content.Text(prompt)))
    }

    fun close() {
        engine?.close()
        engine = null
    }

    companion object {
        private const val MODEL_FILENAME = "gemma-4-E2B-it.litertlm"
        private const val TAG = "GemmaProject"
        fun getGemmaModelFile(context: Context): File? {
            val externalFilesDir = context.getExternalFilesDir(null)
            if (externalFilesDir != null) {
                val modelFile = File(externalFilesDir, MODEL_FILENAME)
                if (modelFile.exists() && modelFile.length() > 0) {
                    return modelFile
                }
            }
            return null
        }
        suspend fun initializeGemmaEngine(context: Context): GemmaEngine? {
            val modelFile = getGemmaModelFile(context)

            if (modelFile == null) {
                Log.e(TAG, "Model file not found. Ensure it was pushed to external storage.")
                return null
            }
            val engineConfig = EngineConfig(
                modelPath = modelFile.absolutePath,
                backend = Backend.GPU(),
                cacheDir = context.cacheDir.path
            )

            val gemmaEngine = create(context, engineConfig)
            try {
                gemmaEngine.initialize()
                Log.i(TAG, "Gemma 4 initialized successfully from local storage!")
                return gemmaEngine
            } catch (e: Exception) {
                Log.e(TAG, "Failed to initialize Gemma engine ", e)
                return null
            }
        }

        fun create(context: Context, config: EngineConfig): GemmaEngine {
            val instance = GemmaEngine(context, config.modelPath)
            instance.customConfig = config
            return instance
        }
    }
}

val Message.text: String
    get() = this.toString()
