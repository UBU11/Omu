package com.example.omu.ml

import android.content.Context
import com.google.ai.edge.litertlm.Backend
import com.google.ai.edge.litertlm.Engine
import com.google.ai.edge.litertlm.EngineConfig
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.flow.Flow
import kotlinx.coroutines.withContext

class GemmaEngine(private val context: Context, private val modelPath: String) {
    private var engine: Engine? = null

    suspend fun initialize() = withContext(Dispatchers.IO) {
        val config = EngineConfig(
            modelPath = modelPath,
            backend = Backend.GPU(),
            cacheDir = context.cacheDir.path
        )
        engine = Engine(config).apply { initialize() }
    }

    fun generateResponse(prompt: String): Flow<String>? {
        val activeEngine = engine ?: return null
        val conversation = activeEngine.createConversation()
        return conversation.sendMessageAsync(prompt)
    }


    fun close() {
        engine?.close()
        engine = null
    }
}
