package com.bam.livekittrainer

import android.content.Context
import org.json.JSONArray
import org.json.JSONObject

class ProjectStore(context: Context) {
    private val prefs = context.getSharedPreferences("wake_word_projects", Context.MODE_PRIVATE)

    fun loadProjects(): List<WakeWordProject> {
        val raw = prefs.getString(KEY_PROJECTS, "[]") ?: "[]"
        val array = JSONArray(raw)
        return buildList {
            for (index in 0 until array.length()) {
                val item = array.getJSONObject(index)
                add(
                    WakeWordProject(
                        id = item.getString("id"),
                        phrase = item.getString("phrase"),
                        slug = item.getString("slug"),
                        createdAtMillis = item.getLong("created_at_millis"),
                    ),
                )
            }
        }
    }

    fun addProject(project: WakeWordProject) {
        val projects = loadProjects().toMutableList()
        projects.add(0, project)
        saveProjects(projects)
    }

    fun loadClips(projectId: String): List<ClipRecord> {
        val raw = prefs.getString(clipsKey(projectId), "[]") ?: "[]"
        val array = JSONArray(raw)
        return buildList {
            for (index in 0 until array.length()) {
                val item = array.getJSONObject(index)
                add(
                    ClipRecord(
                        id = item.getString("id"),
                        projectId = item.getString("project_id"),
                        projectSlug = item.getString("project_slug"),
                        filePath = item.getString("file_path"),
                        label = ClipLabel.valueOf(item.getString("label")),
                        prompt = item.getString("prompt"),
                        spokenPhrase = item.getString("spoken_phrase"),
                        recordedAtMillis = item.getLong("recorded_at_millis"),
                        durationMs = item.getLong("duration_ms"),
                        sampleRateHz = item.getInt("sample_rate_hz"),
                        channels = item.getInt("channels"),
                        encoding = item.getString("encoding"),
                    ),
                )
            }
        }
    }

    fun addClip(clip: ClipRecord) {
        val clips = loadClips(clip.projectId).toMutableList()
        clips.add(0, clip)
        saveClips(clip.projectId, clips)
    }

    fun deleteClip(clip: ClipRecord) {
        saveClips(
            clip.projectId,
            loadClips(clip.projectId).filterNot { it.id == clip.id },
        )
    }

    private fun saveProjects(projects: List<WakeWordProject>) {
        val array = JSONArray()
        projects.forEach { project ->
            array.put(
                JSONObject()
                    .put("id", project.id)
                    .put("phrase", project.phrase)
                    .put("slug", project.slug)
                    .put("created_at_millis", project.createdAtMillis),
            )
        }
        prefs.edit().putString(KEY_PROJECTS, array.toString()).apply()
    }

    private fun saveClips(projectId: String, clips: List<ClipRecord>) {
        val array = JSONArray()
        clips.forEach { clip ->
            array.put(
                JSONObject()
                    .put("id", clip.id)
                    .put("project_id", clip.projectId)
                    .put("project_slug", clip.projectSlug)
                    .put("file_path", clip.filePath)
                    .put("label", clip.label.name)
                    .put("prompt", clip.prompt)
                    .put("spoken_phrase", clip.spokenPhrase)
                    .put("recorded_at_millis", clip.recordedAtMillis)
                    .put("duration_ms", clip.durationMs)
                    .put("sample_rate_hz", clip.sampleRateHz)
                    .put("channels", clip.channels)
                    .put("encoding", clip.encoding),
            )
        }
        prefs.edit().putString(clipsKey(projectId), array.toString()).apply()
    }

    private fun clipsKey(projectId: String): String = "clips_$projectId"

    private companion object {
        const val KEY_PROJECTS = "projects"
    }
}
