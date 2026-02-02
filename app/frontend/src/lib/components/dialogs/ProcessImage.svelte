<script lang="ts">
	const apiBaseUrl = import.meta.env.VITE_API_BASE_URL;
	
	const { student_no, studentItem }: { student_no: string, studentItem: (StudentAnswer & {label: string}) } = $props();

	import type { Student, StudentAnswer } from '$lib/types/types.ts';
	import * as Dialog from '$lib/components/ui/dialog/index.js';
	import { Label } from '$lib/components/ui/label/index.js';

	async function recrop() {
		// TODO: function
	}

	async function sendImage() {
		// TODO: function
	}
</script>


<Dialog.Content class="max-h-[90vh] overflow-y-auto">
	<Dialog.Title>{student_no}: Item {studentItem.label}</Dialog.Title>
	<button class="button-secondary">Send/replace image</button>
	<button class="button-secondary">Recrop</button>
</Dialog.Content>

<!-- TODO: forms, image cropping interface, etc. -->
