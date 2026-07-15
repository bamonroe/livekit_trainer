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

    private companion object {
        const val KEY_PROJECTS = "projects"
    }
}
