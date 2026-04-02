<script lang="ts">
  import * as Sheet from '$lib/components/ui/sheet/index.ts';
</script>



<Sheet.Content side="bottom"
                class="flex max-h-[80dvh] flex-col px-4 py-4">
  <h1>User manual</h1>
  <span class="-mt-6"></span>

  <div class="overflow-y-auto text-sm border-t border-b border-border
              space-y-4 pt-4 pr-2 -mr-2">
    <p class="text-muted-foreground">
      Welcome to <strong>SIPAT.MATH</strong> — an AI-powered grading tool for scanned student
      assessments.
      To serve as a guide for testers, the following expected user flow is devised.
    </p>

    <section>
      <h4 class="font-semibold">Main user flow</h4>
      <ol class="text-muted-foreground list-decimal space-y-0.5 pl-4">
        <li>Register section</li>
        <li>Add students for each section</li>
        <li>Create test instance</li>
        <li>
          Fill up test item details:
          <ol class="list-[lower-roman] pl-6">
            <li>Item type</li>
            <li>Label</li>
            <li>Expected answer / rubric questions</li>
          </ol>
        </li>
        <li>
          Outside of the application…
          <ol class="list-[lower-roman] pl-6">
            <li>Take photos</li>
            <li>[N2H] Rename according to student number</li>
            <li>[N2H] If necessary, rotate orientation</li>
          </ol>
        </li>
        <li>
          Bulk upload photos from gallery
          <ol class="list-[lower-roman] pl-6">
            <li>If need, rename according to student number</li>
            <li>If need, rotate orientation/flip image</li>
          </ol>
        </li>
        <li>Send for segmentation and AI evaluation</li>
        <li>
          Verify results
          <ol class="list-[lower-roman] pl-6">
            <li>If need, recapture/redo/manual crop</li>
          </ol>
        </li>
        <li>Check scores</li>
        <li>If wrong assessment, edit manually</li>
        <li>Download spreadsheet</li>
      </ol>
    </section>

    <section>
      <h4 class="font-semibold">What currently works</h4>
      <ul class="text-muted-foreground list-disc space-y-0.5 pl-4" >
        <li>Background is dark/deeply contrasting with pad paper</li>
        <li>Paper used is Grade 3 pad paper or plain white paper. Multiple pages is allowed.</li>
        <li>
          Answers are marked with dots in corners:
          <ul>
            <li>- Dots are drawn with permanent marker/pen with completely solid dots</li>
            <li>- Thickness is ~4x that of the writings/rules of pad paper</li>
          </ul>
        </li>
      </ul>
    </section>

    <section>
      <h4 class="font-semibold">What has not been tested</h4>
      <ul class="text-muted-foreground list-disc space-y-0.5 pl-4" >
        <li>Background is very bright and blends with contour of pad paper</li>
        <li>Paper is crumpled</li>
        <li>Dots are drawn with pencil</li>
        <li>Thickness is &lt; 4x of writings/rules of pad paper</li>
      </ul>
    </section>
  </div>
</Sheet.Content>
